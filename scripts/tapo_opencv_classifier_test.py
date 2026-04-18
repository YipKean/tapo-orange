import functools
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

import tapo_opencv_test as base
from train_identity_classifier import (
	CLASS_NAMES,
	buildModel,
	resolveCheckpointPretrainedMode,
	resolveDevice,
)


CHECKPOINT_REL_PATH = Path( "artifacts/identity_classifier_pretrained/best.pt" )
DEFAULT_DEVICE = "cuda"
UNKNOWN_MARGIN = 0.06


class ClassifierRuntime:
	def __init__(self) -> None:
		self.repoRoot = Path( __file__ ).resolve().parents[1]
		self.checkpointPath = ( self.repoRoot / CHECKPOINT_REL_PATH ).resolve()
		self.loaded = False
		self.device = "cpu"
		self.imageSize = 224
		self.classNames = list( CLASS_NAMES )
		self.model = None

	def ensureLoaded(self) -> None:
		if self.loaded:
			return
		if not self.checkpointPath.exists():
			raise FileNotFoundError(
				f"Classifier checkpoint not found: {self.checkpointPath}"
			)
		self.device = resolveDevice( DEFAULT_DEVICE )
		checkpoint = torch.load( self.checkpointPath, map_location=self.device )
		self.imageSize = int( checkpoint.get( "image_size", 224 ) )
		self.classNames = checkpoint.get( "class_names", list( CLASS_NAMES ) )
		modelPretrainedMode = resolveCheckpointPretrainedMode( checkpoint )
		self.model, _ = buildModel( modelPretrainedMode )
		self.model.load_state_dict( checkpoint["model_state_dict"] )
		self.model.to( self.device )
		self.model.eval()
		self.loaded = True


classifierRuntime = ClassifierRuntime()


def cropIdentityRoi(
	frame: np.ndarray,
	box: tuple[int, int, int, int],
) -> np.ndarray | None:
	x1, y1, x2, y2 = box
	frameHeight, frameWidth = frame.shape[:2]
	x1 = max( 0, min( x1, frameWidth - 1 ) )
	y1 = max( 0, min( y1, frameHeight - 1 ) )
	x2 = max( x1 + 1, min( x2, frameWidth ) )
	y2 = max( y1 + 1, min( y2, frameHeight ) )
	boxWidth = x2 - x1
	boxHeight = y2 - y1
	insetX = max( 1, int( boxWidth * 0.16 ) )
	insetY = max( 1, int( boxHeight * 0.14 ) )
	rx1 = min( x2 - 1, x1 + insetX )
	ry1 = min( y2 - 1, y1 + insetY )
	rx2 = max( rx1 + 1, x2 - insetX )
	ry2 = max( ry1 + 1, y2 - insetY )
	roi = frame[ry1:ry2, rx1:rx2]
	if roi.size == 0:
		return None
	return roi


def preprocessCrop(crop: np.ndarray, imageSize: int) -> torch.Tensor:
	crop = cv2.cvtColor( crop, cv2.COLOR_BGR2RGB )
	crop = cv2.resize( crop, ( imageSize, imageSize ), interpolation=cv2.INTER_AREA )
	tensor = torch.from_numpy( crop ).permute( 2, 0, 1 ).float() / 255.0
	return ( tensor - 0.5 ) / 0.5


def emptyEvidence() -> base.IdentityEvidence:
	return base.IdentityEvidence(
		orange_ratio=0.0,
		white_ratio=0.0,
		black_ratio=0.0,
		black_blob_count=0,
		torso_core_white_ratio=0.0,
		torso_core_black_ratio=0.0,
		torso_core_black_blob_count=0,
		torso_core_goblin_evidence=0.0,
		periphery_goblin_evidence=0.0,
		blue_spill_ratio=0.0,
		low_light=False,
		orange_evidence=0.0,
		goblin_evidence=0.0,
		contamination_downgraded=False,
	)


def classifyCatIdentity(
	frame: np.ndarray,
	box: tuple[int, int, int, int],
	*,
	contamination_periphery_margin_max: float,
) -> tuple[str, float, base.IdentityEvidence]:
	_ = contamination_periphery_margin_max
	crop = cropIdentityRoi( frame, box )
	if crop is None:
		return "unknown", 0.0, emptyEvidence()

	classifierRuntime.ensureLoaded()
	assert classifierRuntime.model is not None

	with torch.no_grad():
		inputTensor = preprocessCrop( crop, classifierRuntime.imageSize ).unsqueeze( 0 )
		inputTensor = inputTensor.to( classifierRuntime.device )
		logits = classifierRuntime.model( inputTensor )
		probs = torch.softmax( logits, dim=1 ).squeeze( 0 ).detach().cpu().numpy()

	orangeProb = float( probs[0] )
	goblinProb = float( probs[1] )
	predIndex = int( np.argmax( probs ) )
	predLabel = classifierRuntime.classNames[predIndex]
	predConf = float( probs[predIndex] )

	evidence = replace(
		emptyEvidence(),
		orange_evidence=orangeProb,
		goblin_evidence=goblinProb,
		torso_core_goblin_evidence=goblinProb,
		periphery_goblin_evidence=goblinProb,
		torso_core_white_ratio=goblinProb,
	)

	if abs( goblinProb - orangeProb ) < UNKNOWN_MARGIN:
		return "unknown", predConf, evidence
	if predLabel == "goblin":
		return "white_black_dotted", predConf, evidence
	return "orange", predConf, evidence


def isGoblinSupportFrame(
	label: str,
	label_conf: float,
	evidence: base.IdentityEvidence,
	*,
	min_conf: float,
	min_margin: float,
	min_torso_black_blobs: int,
	min_torso_white_ratio: float,
) -> bool:
	_ = min_torso_black_blobs
	_ = min_torso_white_ratio
	if label != "white_black_dotted":
		return False
	return (
		label_conf >= min_conf
		and evidence.goblin_evidence >= evidence.orange_evidence + min_margin
	)


originalBuildModeText = base.build_mode_text


def buildModeText(detectorRuntime: base.DetectorRuntime) -> str:
	return f"{originalBuildModeText( detectorRuntime )} + classifier"


def main() -> int:
	classifierRuntime.ensureLoaded()
	base.classify_cat_identity = classifyCatIdentity
	base.is_goblin_support_frame = isGoblinSupportFrame
	base.build_mode_text = buildModeText
	print(
		f"Loaded classifier checkpoint: {classifierRuntime.checkpointPath} "
		f"({classifierRuntime.device})"
	)
	return base.main()


if __name__ == "__main__":
	raise SystemExit( main() )
