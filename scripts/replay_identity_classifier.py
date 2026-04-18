import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from train_identity_classifier import (
	CLASS_NAMES,
	buildModel,
	resolveCheckpointPretrainedMode,
	resolveDevice,
)


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the trained Orange vs Goblin classifier on a replay clip."
	)
	parser.add_argument(
		"--video",
		required=True,
		help="Video file to replay.",
	)
	parser.add_argument(
		"--checkpoint",
		default="artifacts/identity_classifier/best.pt",
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
		"--sample-every",
		type=int,
		default=3,
		help="Only classify every Nth frame.",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=0,
		help="Optional cap on processed frames. Zero disables the cap.",
	)
	parser.add_argument(
		"--output-csv",
		default="tmp/identity_replay.csv",
		help="CSV path for per-detection probabilities.",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without showing the OpenCV preview window.",
	)
	parser.add_argument(
		"--window-name",
		default="Identity Replay",
		help="Preview window title.",
	)
	parser.add_argument(
		"--preview-width",
		type=int,
		default=1280,
		help="Initial preview window width in pixels.",
	)
	parser.add_argument(
		"--preview-height",
		type=int,
		default=720,
		help="Initial preview window height in pixels.",
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
	return parser.parse_args()


def detectCats(
	detector: YOLO,
	frame: np.ndarray,
	confidenceThreshold: float,
	catClassId: int,
	catImgsz: int,
	device: str,
) -> list[tuple[tuple[int, int, int, int], float]]:
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


def cropIdentityRoi(
	frame: np.ndarray,
	box: tuple[int, int, int, int],
	*,
	torsoInsetX: float,
	torsoInsetY: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
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
		return None, ( rx1, ry1, rx2, ry2 )
	return roi, ( rx1, ry1, rx2, ry2 )


def preprocessCrop(crop: np.ndarray, imageSize: int) -> torch.Tensor:
	crop = cv2.cvtColor( crop, cv2.COLOR_BGR2RGB )
	crop = cv2.resize( crop, ( imageSize, imageSize ), interpolation=cv2.INTER_AREA )
	tensor = torch.from_numpy( crop ).permute( 2, 0, 1 ).float() / 255.0
	return ( tensor - 0.5 ) / 0.5


def predictionColor(predLabel: str) -> tuple[int, int, int]:
	if predLabel == "goblin":
		return ( 32, 196, 32 )
	return ( 0, 140, 255 )


def drawPrediction(
	frame: np.ndarray,
	box: tuple[int, int, int, int],
	torsoBox: tuple[int, int, int, int],
	*,
	catScore: float,
	orangeProb: float,
	goblinProb: float,
	predLabel: str,
	predConf: float,
	detectionIndex: int,
) -> None:
	x1, y1, x2, y2 = box
	tx1, ty1, tx2, ty2 = torsoBox
	color = predictionColor( predLabel )
	cv2.rectangle( frame, ( x1, y1 ), ( x2, y2 ), color, 2 )
	cv2.rectangle( frame, ( tx1, ty1 ), ( tx2, ty2 ), ( 255, 255, 255 ), 1 )
	labelLines = [
		f"#{detectionIndex} {predLabel} {predConf:.2f}",
		f"o={orangeProb:.2f} g={goblinProb:.2f}",
		f"cat={catScore:.2f}",
	]
	textY = max( 18, y1 - 44 )
	for index, line in enumerate( labelLines ):
		lineY = textY + index * 16
		cv2.putText(
			frame,
			line,
			( x1, lineY ),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			( 0, 0, 0 ),
			3,
			cv2.LINE_AA,
		)
		cv2.putText(
			frame,
			line,
			( x1, lineY ),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)


def drawFrameHeader(
	frame: np.ndarray,
	*,
	frameIndex: int,
	processedFrames: int,
	detectionCount: int,
) -> None:
	text = (
		f"frame={frameIndex} processed={processedFrames} "
		f"detections={detectionCount} q=quit"
	)
	cv2.putText(
		frame,
		text,
		( 14, 24 ),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		( 0, 0, 0 ),
		3,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		text,
		( 14, 24 ),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		( 240, 240, 240 ),
		1,
		cv2.LINE_AA,
	)


def main() -> int:
	args = parseArgs()
	repoRoot = Path( __file__ ).resolve().parent.parent
	videoPath = ( repoRoot / args.video ).resolve()
	checkpointPath = ( repoRoot / args.checkpoint ).resolve()
	outputCsvPath = ( repoRoot / args.output_csv ).resolve()
	outputCsvPath.parent.mkdir( parents=True, exist_ok=True )

	if not videoPath.exists():
		raise FileNotFoundError( f"Video not found: {videoPath}" )
	if not checkpointPath.exists():
		raise FileNotFoundError( f"Checkpoint not found: {checkpointPath}" )

	device = resolveDevice( args.device )
	checkpoint = torch.load( checkpointPath, map_location=device )
	imageSize = int( checkpoint.get( "image_size", 224 ) )
	classNames = checkpoint.get( "class_names", list( CLASS_NAMES ) )

	modelPretrainedMode = resolveCheckpointPretrainedMode( checkpoint )
	model, _ = buildModel( modelPretrainedMode )
	model.load_state_dict( checkpoint["model_state_dict"] )
	model.to( device )
	model.eval()

	detector = YOLO( str( ( repoRoot / args.cat_model ).resolve() ) )
	capture = cv2.VideoCapture( str( videoPath ) )
	if not capture.isOpened():
		raise RuntimeError( f"OpenCV could not open video file: {videoPath}" )
	videoFps = float( capture.get( cv2.CAP_PROP_FPS ) )
	if not np.isfinite( videoFps ) or videoFps <= 0:
		videoFps = 30.0
	previewDelayMs = max( 1, int( round( 1000.0 * max( args.sample_every, 1 ) / videoFps ) ) )

	if not args.headless:
		cv2.namedWindow( args.window_name, cv2.WINDOW_NORMAL )
		cv2.resizeWindow( args.window_name, args.preview_width, args.preview_height )

	rows: list[dict[str, str]] = []
	frameIndex = 0
	processedFrames = 0
	quitRequested = False

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
				frame,
				args.cat_confidence,
				args.cat_class_id,
				args.cat_imgsz,
				device,
			)
			previewFrame = frame.copy()

			for detectionIndex, ( box, catScore ) in enumerate( detections ):
				crop, torsoBox = cropIdentityRoi(
					frame,
					box,
					torsoInsetX=args.torso_inset_x,
					torsoInsetY=args.torso_inset_y,
				)
				if crop is None:
					continue
				inputTensor = preprocessCrop( crop, imageSize ).unsqueeze( 0 ).to( device )
				logits = model( inputTensor )
				probs = torch.softmax( logits, dim=1 ).squeeze( 0 ).detach().cpu().numpy()
				predIndex = int( np.argmax( probs ) )
				predLabel = classNames[predIndex]
				predConf = float( probs[predIndex] )
				orangeProb = float( probs[0] )
				goblinProb = float( probs[1] )
				rows.append(
					{
						"frame_index": str( frameIndex ),
						"detection_index": str( detectionIndex ),
						"x1": str( box[0] ),
						"y1": str( box[1] ),
						"x2": str( box[2] ),
						"y2": str( box[3] ),
						"cat_confidence": f"{catScore:.6f}",
						"orange_prob": f"{orangeProb:.6f}",
						"goblin_prob": f"{goblinProb:.6f}",
						"pred_label": predLabel,
						"pred_confidence": f"{predConf:.6f}",
					}
				)
				drawPrediction(
					previewFrame,
					box,
					torsoBox,
					catScore=catScore,
					orangeProb=orangeProb,
					goblinProb=goblinProb,
					predLabel=predLabel,
					predConf=predConf,
					detectionIndex=detectionIndex,
				)

			processedFrames += 1
			if not args.headless:
				drawFrameHeader(
					previewFrame,
					frameIndex=frameIndex,
					processedFrames=processedFrames,
					detectionCount=len( detections ),
				)
				cv2.imshow( args.window_name, previewFrame )
				key = cv2.waitKey( previewDelayMs ) & 0xFF
				if key == ord( "q" ):
					quitRequested = True
					break
			frameIndex += 1
			if args.max_frames > 0 and processedFrames >= args.max_frames:
				break

	capture.release()
	if not args.headless:
		cv2.destroyWindow( args.window_name )

	with outputCsvPath.open( "w", encoding="utf-8", newline="" ) as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"frame_index",
				"detection_index",
				"x1",
				"y1",
				"x2",
				"y2",
				"cat_confidence",
				"orange_prob",
				"goblin_prob",
				"pred_label",
				"pred_confidence",
			],
		)
		writer.writeheader()
		writer.writerows( rows )

	print( f"video={videoPath}" )
	print( f"output_csv={outputCsvPath}" )
	print( f"detections={len( rows )}" )
	if quitRequested:
		print( "stopped=quit_key" )
	return 0


if __name__ == "__main__":
	raise SystemExit( main() )
