import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from identity_dataset_builder import (
	VIDEO_SUFFIXES,
	configureDetectorDevice,
	cropIdentityRoi,
	detectCats,
	loadDetector,
	stableSplit,
)
from train_identity_classifier import (
	CLASS_NAMES,
	buildModel,
	resolveCheckpointPretrainedMode,
	resolveDevice,
)


@dataclass
class SortedCropRecord:
	sourcePath: str
	split: str
	label: str
	frameIndex: int
	detectionIndex: int
	catConfidence: float
	orangeProb: float
	goblinProb: float
	predConfidence: float
	cropPath: str
	width: int
	height: int


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Classify cat crops from a mixed video into provisional train/val "
			"Orange and Goblin folders for manual cleanup."
		)
	)
	inputGroup = parser.add_mutually_exclusive_group( required=True )
	inputGroup.add_argument(
		"--video",
		help="Video file to sort.",
	)
	inputGroup.add_argument(
		"--input-dir",
		help="Directory of videos to sort. Each video gets its own output subfolder.",
	)
	parser.add_argument(
		"--output-dir",
		default="datasets/sorted_identity",
		help="Output directory containing one subfolder per input video.",
	)
	parser.add_argument(
		"--checkpoint",
		default="artifacts/identity_classifier_pretrained/best.pt",
		help="Classifier checkpoint path.",
	)
	parser.add_argument(
		"--cat-model",
		default="models/yolov8m.pt",
		help="YOLO model used to detect cat boxes.",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="Inference device preference: cuda or cpu.",
	)
	parser.add_argument(
		"--sample-every",
		type=int,
		default=5,
		help="Only process every Nth frame.",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=0,
		help="Optional cap on processed frames. Zero disables the cap.",
	)
	parser.add_argument(
		"--cat-confidence",
		type=float,
		default=0.08,
		help="Minimum cat detection confidence.",
	)
	parser.add_argument(
		"--cat-class-id",
		type=int,
		default=15,
		help="COCO class id for cat.",
	)
	parser.add_argument(
		"--cat-imgsz",
		type=int,
		default=1920,
		help="YOLO inference image size.",
	)
	parser.add_argument(
		"--min-box-size",
		type=int,
		default=96,
		help="Minimum cat box width and height to keep.",
	)
	parser.add_argument(
		"--torso-inset-x",
		type=float,
		default=0.16,
		help="Horizontal inset ratio for torso crop.",
	)
	parser.add_argument(
		"--torso-inset-y",
		type=float,
		default=0.14,
		help="Vertical inset ratio for torso crop.",
	)
	parser.add_argument(
		"--train-ratio",
		type=float,
		default=0.8,
		help="Stable split ratio for saved crops.",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=250,
		help="Print progress every N processed frames. Zero disables progress output.",
	)
	return parser.parse_args()


def preprocessCrop(crop: np.ndarray, imageSize: int) -> torch.Tensor:
	crop = cv2.cvtColor( crop, cv2.COLOR_BGR2RGB )
	crop = cv2.resize( crop, ( imageSize, imageSize ), interpolation=cv2.INTER_AREA )
	tensor = torch.from_numpy( crop ).permute( 2, 0, 1 ).float() / 255.0
	return ( tensor - 0.5 ) / 0.5


def writeManifest(manifestPath: Path, records: list[SortedCropRecord]) -> None:
	with manifestPath.open( "w", encoding="utf-8", newline="" ) as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"source_path",
				"split",
				"label",
				"frame_index",
				"detection_index",
				"cat_confidence",
				"orange_prob",
				"goblin_prob",
				"pred_confidence",
				"crop_path",
				"width",
				"height",
			],
		)
		writer.writeheader()
		for record in records:
			writer.writerow(
				{
					"source_path": record.sourcePath,
					"split": record.split,
					"label": record.label,
					"frame_index": record.frameIndex,
					"detection_index": record.detectionIndex,
					"cat_confidence": f"{record.catConfidence:.6f}",
					"orange_prob": f"{record.orangeProb:.6f}",
					"goblin_prob": f"{record.goblinProb:.6f}",
					"pred_confidence": f"{record.predConfidence:.6f}",
					"crop_path": record.cropPath,
					"width": record.width,
					"height": record.height,
				}
			)


def writeSummary(
	summaryPath: Path,
	*,
	videoPath: Path,
	checkpointPath: Path,
	detectorDevice: str,
	classifierDevice: str,
	sampleEvery: int,
	processedFrames: int,
	savedCrops: int,
	perLabelCounts: dict[str, int],
	stoppedReason: str,
) -> None:
	with summaryPath.open( "w", encoding="utf-8" ) as handle:
		handle.write( f"video={videoPath}\n" )
		handle.write( f"checkpoint={checkpointPath}\n" )
		handle.write( f"detector_device={detectorDevice}\n" )
		handle.write( f"classifier_device={classifierDevice}\n" )
		handle.write( f"sample_every={sampleEvery}\n" )
		handle.write( f"processed_frames={processedFrames}\n" )
		handle.write( f"saved_crops={savedCrops}\n" )
		handle.write( f"stopped_reason={stoppedReason}\n" )
		for className in CLASS_NAMES:
			handle.write( f"{className}_crops={perLabelCounts[className]}\n" )


def resolveVideoPaths(repoRoot: Path, args: argparse.Namespace) -> list[Path]:
	if args.video:
		videoPath = ( repoRoot / args.video ).resolve()
		if not videoPath.exists():
			raise FileNotFoundError( f"Video not found: {videoPath}" )
		return [videoPath]

	inputDir = ( repoRoot / args.input_dir ).resolve()
	if not inputDir.exists():
		raise FileNotFoundError( f"Input directory not found: {inputDir}" )
	if not inputDir.is_dir():
		raise NotADirectoryError( f"Input path is not a directory: {inputDir}" )

	videoPaths = sorted(
		path for path in inputDir.iterdir()
		if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
	)
	if not videoPaths:
		raise FileNotFoundError( f"No video files found in: {inputDir}" )
	return videoPaths


def processVideo(
	*,
	args: argparse.Namespace,
	repoRoot: Path,
	videoPath: Path,
	outputRootDir: Path,
	checkpointPath: Path,
	model: torch.nn.Module,
	classNames: list[str],
	imageSize: int,
	detector: object,
	detectorBackend: str,
	detectorDevice: str,
	classifierDevice: str,
) -> int:
	outputDir = outputRootDir / videoPath.stem
	manifestPath = outputDir / "manifest.csv"
	summaryPath = outputDir / "summary.txt"

	outputDir.mkdir( parents=True, exist_ok=True )
	for split in ( "train", "val" ):
		for className in CLASS_NAMES:
			( outputDir / split / className ).mkdir( parents=True, exist_ok=True )

	capture = cv2.VideoCapture( str( videoPath ) )
	if not capture.isOpened():
		raise RuntimeError( f"OpenCV could not open video file: {videoPath}" )

	sourcePath = videoPath.relative_to( repoRoot ).as_posix()
	records: list[SortedCropRecord] = []
	perLabelCounts = {className: 0 for className in CLASS_NAMES}
	frameIndex = 0
	processedFrames = 0
	savedCrops = 0
	stoppedReason = "completed"

	try:
		with torch.no_grad():
			while True:
				ok, frame = capture.read()
				if not ok:
					break
				if args.sample_every > 1 and frameIndex % args.sample_every != 0:
					frameIndex += 1
					continue

				detections = detectCats(
					detector,
					detectorBackend,
					detectorDevice,
					frame,
					args.cat_confidence,
					args.cat_class_id,
					args.cat_imgsz,
				)
				for detectionIndex, ( box, catScore ) in enumerate( detections ):
					boxWidth = box[2] - box[0]
					boxHeight = box[3] - box[1]
					if boxWidth < args.min_box_size or boxHeight < args.min_box_size:
						continue

					crop = cropIdentityRoi(
						frame,
						box,
						torsoInsetX=args.torso_inset_x,
						torsoInsetY=args.torso_inset_y,
					)
					if crop is None:
						continue

					inputTensor = preprocessCrop( crop, imageSize ).unsqueeze( 0 ).to( classifierDevice )
					logits = model( inputTensor )
					probs = torch.softmax( logits, dim=1 ).squeeze( 0 ).detach().cpu().numpy()
					predIndex = int( np.argmax( probs ) )
					predLabel = classNames[predIndex]
					predConf = float( probs[predIndex] )
					orangeProb = float( probs[0] )
					goblinProb = float( probs[1] )
					splitKey = stableSplit(
						f"{sourcePath}:{frameIndex}:{detectionIndex}:{predLabel}",
						args.train_ratio,
					)
					destinationDir = outputDir / splitKey / predLabel
					cropName = f"{videoPath.stem}_frame{frameIndex:06d}_det{detectionIndex:02d}.jpg"
					cropPath = destinationDir / cropName
					cv2.imwrite( str( cropPath ), crop )
					records.append(
						SortedCropRecord(
							sourcePath=sourcePath,
							split=splitKey,
							label=predLabel,
							frameIndex=frameIndex,
							detectionIndex=detectionIndex,
							catConfidence=float( catScore ),
							orangeProb=orangeProb,
							goblinProb=goblinProb,
							predConfidence=predConf,
							cropPath=cropPath.relative_to( outputDir ).as_posix(),
							width=crop.shape[1],
							height=crop.shape[0],
						)
					)
					perLabelCounts[predLabel] += 1
					savedCrops += 1

				frameIndex += 1
				processedFrames += 1
				if args.progress_every > 0 and processedFrames % args.progress_every == 0:
					print(
						f"processed_frames={processedFrames} "
						f"saved_crops={savedCrops} "
						f"orange={perLabelCounts['orange']} "
						f"goblin={perLabelCounts['goblin']}"
					)
				if args.max_frames > 0 and processedFrames >= args.max_frames:
					stoppedReason = "max_frames"
					break
	except KeyboardInterrupt:
		stoppedReason = "keyboard_interrupt"
		print( "Stopped early by keyboard interrupt. Writing partial manifest and summary." )
	finally:
		capture.release()
		writeManifest( manifestPath, records )
		writeSummary(
			summaryPath,
			videoPath=videoPath,
			checkpointPath=checkpointPath,
			detectorDevice=detectorDevice,
			classifierDevice=classifierDevice,
			sampleEvery=args.sample_every,
			processedFrames=processedFrames,
			savedCrops=savedCrops,
			perLabelCounts=perLabelCounts,
			stoppedReason=stoppedReason,
		)

	print( f"video={videoPath}" )
	print( f"output_dir={outputDir}" )
	print( f"manifest={manifestPath}" )
	print( f"processed_frames={processedFrames}" )
	print( f"saved_crops={savedCrops}" )
	for className in CLASS_NAMES:
		print( f"{className}={perLabelCounts[className]}" )
	return savedCrops


def main() -> int:
	args = parseArgs()
	repoRoot = Path( __file__ ).resolve().parent.parent
	outputRootDir = ( repoRoot / args.output_dir ).resolve()
	checkpointPath = ( repoRoot / args.checkpoint ).resolve()
	if not checkpointPath.exists():
		raise FileNotFoundError( f"Checkpoint not found: {checkpointPath}" )

	videoPaths = resolveVideoPaths( repoRoot, args )
	outputRootDir.mkdir( parents=True, exist_ok=True )

	classifierDevice = resolveDevice( args.device )
	checkpoint = torch.load( checkpointPath, map_location=classifierDevice )
	imageSize = int( checkpoint.get( "image_size", 224 ) )
	classNames = checkpoint.get( "class_names", list( CLASS_NAMES ) )
	modelPretrainedMode = resolveCheckpointPretrainedMode( checkpoint )
	model, _ = buildModel( modelPretrainedMode )
	model.load_state_dict( checkpoint["model_state_dict"] )
	model.to( classifierDevice )
	model.eval()

	detector, detectorBackend = loadDetector( str( ( repoRoot / args.cat_model ).resolve() ) )
	detectorDevice = configureDetectorDevice( detector, detectorBackend, args.device )

	totalSavedCrops = 0
	for videoIndex, videoPath in enumerate( videoPaths, start=1 ):
		print( f"[{videoIndex}/{len( videoPaths )}] sorting {videoPath.name}" )
		totalSavedCrops += processVideo(
			args=args,
			repoRoot=repoRoot,
			videoPath=videoPath,
			outputRootDir=outputRootDir,
			checkpointPath=checkpointPath,
			model=model,
			classNames=classNames,
			imageSize=imageSize,
			detector=detector,
			detectorBackend=detectorBackend,
			detectorDevice=detectorDevice,
			classifierDevice=classifierDevice,
		)

	print( f"videos_processed={len( videoPaths )}" )
	print( f"total_saved_crops={totalSavedCrops}" )
	return 0


if __name__ == "__main__":
	raise SystemExit( main() )
