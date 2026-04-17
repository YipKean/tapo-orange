import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
CLASS_NAMES = ( "orange", "goblin" )


@dataclass
class CropRecord:
	sourcePath: str
	sourceKind: str
	label: str
	split: str
	cropPath: str
	frameIndex: int
	detectionIndex: int
	confidence: float
	width: int
	height: int


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build a labeled Orange vs Goblin crop dataset from captures/."
	)
	parser.add_argument(
		"--input-dir",
		default="captures",
		help="Directory containing source images and videos.",
	)
	parser.add_argument(
		"--output-dir",
		default="datasets/identity",
		help="Output directory for extracted crops and manifests.",
	)
	parser.add_argument(
		"--cat-model",
		default="models/yolov8m.pt",
		help="YOLO model used to detect cat boxes before cropping.",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="Detector device preference: cuda or cpu.",
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
		help="Inference image size for YOLO .pt models.",
	)
	parser.add_argument(
		"--sample-every",
		type=int,
		default=3,
		help="Sample one frame every N frames for videos.",
	)
	parser.add_argument(
		"--max-samples-per-video",
		type=int,
		default=0,
		help="Optional cap on saved crops per video. Zero disables the cap.",
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
		help="Horizontal inset ratio for the tighter identity crop.",
	)
	parser.add_argument(
		"--torso-inset-y",
		type=float,
		default=0.14,
		help="Vertical inset ratio for the tighter identity crop.",
	)
	parser.add_argument(
		"--train-ratio",
		type=float,
		default=0.8,
		help="Stable source-level train split ratio.",
	)
	parser.add_argument(
		"--labels-csv",
		help=(
			"Optional CSV with columns relative_path,label. "
			"Used for files that cannot be labeled from the filename alone."
		),
	)
	parser.add_argument(
		"--emit-label-template",
		help=(
			"Optional CSV path. Writes one row per media file with any inferred label "
			"so unlabeled files can be filled in manually."
		),
	)
	return parser.parse_args()


def loadDetector(modelPath: str) -> tuple[object, str]:
	suffix = Path( modelPath ).suffix.lower()
	if suffix == ".pt":
		return YOLO( modelPath ), "ultralytics"
	if suffix == ".onnx":
		return cv2.dnn.readNetFromONNX( modelPath ), "opencv_dnn"
	raise ValueError( f"Unsupported model format: {suffix}" )


def configureDetectorDevice(detector: object, backend: str, requestedDevice: str) -> str:
	if backend == "ultralytics":
		if requestedDevice == "cuda":
			if torch.cuda.is_available():
				return "cuda:0"
			print(
				"CUDA was requested but PyTorch CUDA is unavailable. Falling back to CPU.",
				file=sys.stderr,
			)
		return "cpu"

	if requestedDevice == "cuda":
		try:
			detector = detector  # type: ignore[assignment]
			detector.setPreferableBackend( cv2.dnn.DNN_BACKEND_CUDA )
			detector.setPreferableTarget( cv2.dnn.DNN_TARGET_CUDA )
			return "cuda"
		except cv2.error:
			print(
				"CUDA backend is not available in this OpenCV build. Falling back to CPU.",
				file=sys.stderr,
			)

	detector = detector  # type: ignore[assignment]
	detector.setPreferableBackend( cv2.dnn.DNN_BACKEND_OPENCV )
	detector.setPreferableTarget( cv2.dnn.DNN_TARGET_CPU )
	return "cpu"


def detectCats(
	detector: object,
	backend: str,
	device: str,
	frame: np.ndarray,
	confidenceThreshold: float,
	catClassId: int,
	catImgsz: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
	if backend == "ultralytics":
		results = detector.predict(
			source=frame,
			conf=confidenceThreshold,
			classes=[catClassId],
			device=device,
			verbose=False,
			imgsz=catImgsz,
		)
		detections: list[tuple[tuple[int, int, int, int], float]] = []
		if not results or results[0].boxes is None:
			return detections
		xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
		confs = results[0].boxes.conf.detach().cpu().numpy()
		for box, score in zip( xyxy, confs ):
			x1, y1, x2, y2 = [int( value ) for value in box]
			detections.append( ( ( x1, y1, x2, y2 ), float( score ) ) )
		return detections

	detector = detector  # type: ignore[assignment]
	inputSize = 640
	blob = cv2.dnn.blobFromImage(
		frame,
		scalefactor=1 / 255.0,
		size=( inputSize, inputSize ),
		swapRB=True,
		crop=False,
	)
	detector.setInput( blob )
	outputs = detector.forward()
	predictions = np.squeeze( outputs )
	if predictions.ndim != 2:
		return []
	if predictions.shape[0] < predictions.shape[1]:
		predictions = predictions.T
	if predictions.shape[1] < 5 + catClassId:
		return []

	frameHeight, frameWidth = frame.shape[:2]
	scaleX = frameWidth / inputSize
	scaleY = frameHeight / inputSize
	boxes: list[list[int]] = []
	scores: list[float] = []

	for row in predictions:
		if row.shape[0] <= 4 + catClassId:
			continue
		classScores = row[4:]
		classId = int( np.argmax( classScores ) )
		score = float( classScores[classId] )
		if classId != catClassId or score < confidenceThreshold:
			continue
		cx, cy, width, height = row[:4]
		x1 = int( ( cx - width / 2 ) * scaleX )
		y1 = int( ( cy - height / 2 ) * scaleY )
		w = int( width * scaleX )
		h = int( height * scaleY )
		boxes.append( [x1, y1, w, h] )
		scores.append( score )

	if not boxes:
		return []

	indexes = cv2.dnn.NMSBoxes( boxes, scores, confidenceThreshold, 0.45 )
	if len( indexes ) == 0:
		return []

	detections: list[tuple[tuple[int, int, int, int], float]] = []
	for idx in np.array( indexes ).flatten():
		x, y, w, h = boxes[int( idx )]
		detections.append( ( ( x, y, x + w, y + h ), scores[int( idx )] ) )
	return detections


def cropIdentityRoi(
	frame: np.ndarray,
	box: tuple[int, int, int, int],
	*,
	torsoInsetX: float,
	torsoInsetY: float,
) -> np.ndarray | None:
	x1, y1, x2, y2 = box
	frameHeight, frameWidth = frame.shape[:2]
	x1 = max( 0, min( x1, frameWidth - 1 ) )
	y1 = max( 0, min( y1, frameHeight - 1 ) )
	x2 = max( x1 + 1, min( x2, frameWidth ) )
	y2 = max( y1 + 1, min( y2, frameHeight ) )
	boxWidth = x2 - x1
	boxHeight = y2 - y1
	insetX = max( 1, int( boxWidth * torsoInsetX ) )
	insetY = max( 1, int( boxHeight * torsoInsetY ) )
	rx1 = min( x2 - 1, x1 + insetX )
	ry1 = min( y2 - 1, y1 + insetY )
	rx2 = max( rx1 + 1, x2 - insetX )
	ry2 = max( ry1 + 1, y2 - insetY )
	roi = frame[ry1:ry2, rx1:rx2]
	if roi.size == 0:
		return None
	return roi


def inferLabel(relativePath: str, explicitLabels: dict[str, str]) -> str:
	normalizedPath = relativePath.replace( "\\", "/" ).lower()
	if normalizedPath in explicitLabels:
		return explicitLabels[normalizedPath]

	name = Path( normalizedPath ).name
	if "goblin" in normalizedPath and "orange" not in normalizedPath:
		return "goblin"
	if "orange" in normalizedPath and "goblin" not in normalizedPath:
		return "orange"
	if "_goblin." in name or "alert_goblin_" in name or name.endswith( "_goblin.mp4" ):
		return "goblin"
	if "_orange." in name or "alert_orange_" in name or name.endswith( "_orange.mp4" ):
		return "orange"
	return ""


def loadExplicitLabels(labelsCsvPath: str | None) -> dict[str, str]:
	if not labelsCsvPath:
		return {}
	labels: dict[str, str] = {}
	with Path( labelsCsvPath ).open( "r", encoding="utf-8-sig", newline="" ) as handle:
		reader = csv.DictReader( handle )
		for row in reader:
			relativePath = ( row.get( "relative_path", "" ) or "" ).strip().replace( "\\", "/" )
			label = ( row.get( "label", "" ) or "" ).strip().lower()
			if not relativePath or label not in CLASS_NAMES:
				continue
			labels[relativePath.lower()] = label
	return labels


def stableSplit(relativePath: str, trainRatio: float) -> str:
	digest = hashlib.sha1( relativePath.lower().encode( "utf-8" ) ).hexdigest()
	value = int( digest[:8], 16 ) / 0xFFFFFFFF
	return "train" if value < trainRatio else "val"


def iterMediaFiles(inputDir: Path) -> list[Path]:
	mediaFiles: list[Path] = []
	for path in inputDir.rglob( "*" ):
		if not path.is_file():
			continue
		suffix = path.suffix.lower()
		if suffix in IMAGE_SUFFIXES or suffix in VIDEO_SUFFIXES:
			mediaFiles.append( path )
	return sorted( mediaFiles )


def emitLabelTemplate(
	mediaFiles: list[Path],
	inputDir: Path,
	outputPath: Path,
	explicitLabels: dict[str, str],
) -> None:
	outputPath.parent.mkdir( parents=True, exist_ok=True )
	with outputPath.open( "w", encoding="utf-8", newline="" ) as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=["relative_path", "kind", "label"],
		)
		writer.writeheader()
		for mediaPath in mediaFiles:
			relativePath = mediaPath.relative_to( inputDir ).as_posix()
			writer.writerow(
				{
					"relative_path": relativePath,
					"kind": "video" if mediaPath.suffix.lower() in VIDEO_SUFFIXES else "image",
					"label": inferLabel( relativePath, explicitLabels ),
				}
			)


def writeManifest(manifestPath: Path, records: list[CropRecord]) -> None:
	manifestPath.parent.mkdir( parents=True, exist_ok=True )
	with manifestPath.open( "w", encoding="utf-8", newline="" ) as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"source_path",
				"source_kind",
				"label",
				"split",
				"crop_path",
				"frame_index",
				"detection_index",
				"confidence",
				"width",
				"height",
			],
		)
		writer.writeheader()
		for record in records:
			writer.writerow(
				{
					"source_path": record.sourcePath,
					"source_kind": record.sourceKind,
					"label": record.label,
					"split": record.split,
					"crop_path": record.cropPath,
					"frame_index": record.frameIndex,
					"detection_index": record.detectionIndex,
					"confidence": f"{record.confidence:.6f}",
					"width": record.width,
					"height": record.height,
				}
			)


def buildDataset(args: argparse.Namespace) -> int:
	repoRoot = Path( __file__ ).resolve().parent.parent
	inputDir = ( repoRoot / args.input_dir ).resolve()
	outputDir = ( repoRoot / args.output_dir ).resolve()
	cropsDir = outputDir / "crops"
	manifestPath = outputDir / "manifest.csv"
	summaryPath = outputDir / "summary.txt"
	labels = loadExplicitLabels( args.labels_csv )

	if not inputDir.exists():
		raise FileNotFoundError( f"Input directory not found: {inputDir}" )

	mediaFiles = iterMediaFiles( inputDir )
	if args.emit_label_template:
		emitLabelTemplate(
			mediaFiles,
			inputDir,
			( repoRoot / args.emit_label_template ).resolve(),
			labels,
		)

	detector, backend = loadDetector( str( ( repoRoot / args.cat_model ).resolve() ) )
	detectorDevice = configureDetectorDevice( detector, backend, args.device )

	outputDir.mkdir( parents=True, exist_ok=True )
	cropsDir.mkdir( parents=True, exist_ok=True )

	records: list[CropRecord] = []
	skippedUnlabeled: list[str] = []
	skippedNoDetection: list[str] = []
	perLabelCounts = {className: 0 for className in CLASS_NAMES}

	for mediaPath in mediaFiles:
		relativePath = mediaPath.relative_to( inputDir ).as_posix()
		label = inferLabel( relativePath, labels )
		if label not in CLASS_NAMES:
			skippedUnlabeled.append( relativePath )
			continue

		split = stableSplit( relativePath, args.train_ratio )
		if mediaPath.suffix.lower() in IMAGE_SUFFIXES:
			frame = cv2.imread( str( mediaPath ) )
			if frame is None:
				skippedNoDetection.append( relativePath )
				continue
			detections = detectCats(
				detector,
				backend,
				detectorDevice,
				frame,
				args.cat_confidence,
				args.cat_class_id,
				args.cat_imgsz,
			)
			if not detections:
				skippedNoDetection.append( relativePath )
				continue
			bestBox, bestScore = max( detections, key=lambda item: item[1] )
			width = bestBox[2] - bestBox[0]
			height = bestBox[3] - bestBox[1]
			if width < args.min_box_size or height < args.min_box_size:
				skippedNoDetection.append( relativePath )
				continue
			crop = cropIdentityRoi(
				frame,
				bestBox,
				torsoInsetX=args.torso_inset_x,
				torsoInsetY=args.torso_inset_y,
			)
			if crop is None:
				skippedNoDetection.append( relativePath )
				continue
			destinationDir = cropsDir / split / label
			destinationDir.mkdir( parents=True, exist_ok=True )
			cropName = f"{mediaPath.stem}_frame000000_det00.jpg"
			cropPath = destinationDir / cropName
			cv2.imwrite( str( cropPath ), crop )
			records.append(
				CropRecord(
					sourcePath=relativePath,
					sourceKind="image",
					label=label,
					split=split,
					cropPath=cropPath.relative_to( outputDir ).as_posix(),
					frameIndex=0,
					detectionIndex=0,
					confidence=bestScore,
					width=crop.shape[1],
					height=crop.shape[0],
				)
			)
			perLabelCounts[label] += 1
			continue

		capture = cv2.VideoCapture( str( mediaPath ) )
		if not capture.isOpened():
			skippedNoDetection.append( relativePath )
			continue

		frameIndex = 0
		savedForVideo = 0
		detectionCounter = 0
		while True:
			ok, frame = capture.read()
			if not ok:
				break
			if args.sample_every > 1 and frameIndex % args.sample_every != 0:
				frameIndex += 1
				continue

			detections = detectCats(
				detector,
				backend,
				detectorDevice,
				frame,
				args.cat_confidence,
				args.cat_class_id,
				args.cat_imgsz,
			)
			for box, score in detections:
				width = box[2] - box[0]
				height = box[3] - box[1]
				if width < args.min_box_size or height < args.min_box_size:
					continue
				crop = cropIdentityRoi(
					frame,
					box,
					torsoInsetX=args.torso_inset_x,
					torsoInsetY=args.torso_inset_y,
				)
				if crop is None:
					continue
				destinationDir = cropsDir / split / label
				destinationDir.mkdir( parents=True, exist_ok=True )
				cropName = (
					f"{mediaPath.stem}_frame{frameIndex:06d}_det{detectionCounter:02d}.jpg"
				)
				cropPath = destinationDir / cropName
				cv2.imwrite( str( cropPath ), crop )
				records.append(
					CropRecord(
						sourcePath=relativePath,
						sourceKind="video",
						label=label,
						split=split,
						cropPath=cropPath.relative_to( outputDir ).as_posix(),
						frameIndex=frameIndex,
						detectionIndex=detectionCounter,
						confidence=score,
						width=crop.shape[1],
						height=crop.shape[0],
					)
				)
				perLabelCounts[label] += 1
				savedForVideo += 1
				detectionCounter += 1
				if args.max_samples_per_video > 0 and savedForVideo >= args.max_samples_per_video:
					break
			frameIndex += 1
			if args.max_samples_per_video > 0 and savedForVideo >= args.max_samples_per_video:
				break
		capture.release()
		if savedForVideo == 0:
			skippedNoDetection.append( relativePath )

	writeManifest( manifestPath, records )
	with summaryPath.open( "w", encoding="utf-8" ) as handle:
		handle.write( f"input_dir={inputDir}\n" )
		handle.write( f"output_dir={outputDir}\n" )
		handle.write( f"detector_device={detectorDevice}\n" )
		handle.write( f"records={len( records )}\n" )
		for className in CLASS_NAMES:
			handle.write( f"{className}_crops={perLabelCounts[className]}\n" )
		handle.write( f"skipped_unlabeled={len( skippedUnlabeled )}\n" )
		handle.write( f"skipped_no_detection={len( skippedNoDetection )}\n" )
		if skippedUnlabeled:
			handle.write( "unlabeled_files=\n" )
			for item in skippedUnlabeled:
				handle.write( f"{item}\n" )
		if skippedNoDetection:
			handle.write( "no_detection_files=\n" )
			for item in skippedNoDetection:
				handle.write( f"{item}\n" )

	print( f"Wrote manifest: {manifestPath}" )
	print( f"Saved crops: {len( records )}" )
	for className in CLASS_NAMES:
		print( f"{className}: {perLabelCounts[className]}" )
	print( f"Skipped unlabeled: {len( skippedUnlabeled )}" )
	print( f"Skipped no detection: {len( skippedNoDetection )}" )
	return 0


def main() -> int:
	args = parseArgs()
	return buildDataset( args )


if __name__ == "__main__":
	raise SystemExit( main() )
