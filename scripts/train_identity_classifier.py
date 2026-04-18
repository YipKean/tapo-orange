import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


CLASS_NAMES = ( "orange", "goblin" )
CLASS_TO_INDEX = {name: idx for idx, name in enumerate( CLASS_NAMES )}


@dataclass
class Sample:
	imagePath: Path
	label: int
	split: str
	sourcePath: str


class IdentityDataset(Dataset):
	def __init__(self, samples: list[Sample], imageSize: int) -> None:
		self.samples = samples
		self.imageSize = imageSize

	def __len__(self) -> int:
		return len( self.samples )

	def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
		sample = self.samples[index]
		image = cv2.imread( str( sample.imagePath ) )
		if image is None:
			raise FileNotFoundError( f"Unable to read image: {sample.imagePath}" )
		image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
		image = cv2.resize(
			image,
			( self.imageSize, self.imageSize ),
			interpolation=cv2.INTER_AREA,
		)
		tensor = torch.from_numpy( image ).permute( 2, 0, 1 ).float() / 255.0
		tensor = ( tensor - 0.5 ) / 0.5
		return tensor, sample.label


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train a small Orange vs Goblin classifier from extracted crops."
	)
	parser.add_argument(
		"--dataset-dir",
		default="datasets/identity",
		help="Directory containing manifest.csv and extracted crops.",
	)
	parser.add_argument(
		"--manifest",
		default="manifest.csv",
		help="Manifest filename inside dataset-dir.",
	)
	parser.add_argument(
		"--dataset-source",
		choices=( "auto", "folders", "manifest" ),
		default="auto",
		help=(
			"How to load samples. auto prefers crops/train|val folders and falls back "
			"to the manifest."
		),
	)
	parser.add_argument(
		"--output-dir",
		default="artifacts/identity_classifier",
		help="Directory for checkpoints and metrics.",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="Training device preference: cuda or cpu.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=12,
		help="Number of training epochs.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Mini-batch size.",
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=0.001,
		help="AdamW learning rate.",
	)
	parser.add_argument(
		"--weight-decay",
		type=float,
		default=0.0001,
		help="AdamW weight decay.",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=224,
		help="Square input size for the classifier.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=0,
		help="DataLoader worker count.",
	)
	parser.add_argument(
		"--pretrained",
		choices=( "imagenet", "auto", "none" ),
		default="imagenet",
		help=(
			"Backbone initialization. imagenet uses torchvision pretrained weights, "
			"auto falls back to random init if pretrained weights are unavailable."
		),
	)
	parser.add_argument(
		"--selection-metric",
		choices=( "val_accuracy", "val_goblin_recall", "val_goblin_f1" ),
		default="val_goblin_recall",
		help="Metric used to choose the best checkpoint.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=1337,
		help="Random seed.",
	)
	return parser.parse_args()


def setSeed(seed: int) -> None:
	random.seed( seed )
	torch.manual_seed( seed )
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all( seed )


def resolveDevice(requestedDevice: str) -> str:
	if requestedDevice == "cuda" and torch.cuda.is_available():
		return "cuda"
	return "cpu"


def loadSamples(datasetDir: Path, manifestName: str) -> list[Sample]:
	manifestPath = datasetDir / manifestName
	if not manifestPath.exists():
		raise FileNotFoundError( f"Manifest not found: {manifestPath}" )

	samples: list[Sample] = []
	with manifestPath.open( "r", encoding="utf-8-sig", newline="" ) as handle:
		reader = csv.DictReader( handle )
		for row in reader:
			labelName = ( row.get( "label", "" ) or "" ).strip().lower()
			split = ( row.get( "split", "" ) or "" ).strip().lower()
			cropPath = ( row.get( "crop_path", "" ) or "" ).strip()
			sourcePath = ( row.get( "source_path", "" ) or "" ).strip()
			if labelName not in CLASS_TO_INDEX or split not in {"train", "val"} or not cropPath:
				continue
			samples.append(
				Sample(
					imagePath=( datasetDir / cropPath ).resolve(),
					label=CLASS_TO_INDEX[labelName],
					split=split,
					sourcePath=sourcePath,
				)
			)
	return samples


def loadSamplesFromFolders(datasetDir: Path) -> list[Sample]:
	cropsDir = datasetDir / "crops"
	if not cropsDir.exists():
		raise FileNotFoundError( f"Crops directory not found: {cropsDir}" )

	samples: list[Sample] = []
	for split in ( "train", "val" ):
		for className in CLASS_NAMES:
			classDir = cropsDir / split / className
			if not classDir.exists():
				continue
			for imagePath in sorted( classDir.glob( "*.jpg" ) ):
				samples.append(
					Sample(
						imagePath=imagePath.resolve(),
						label=CLASS_TO_INDEX[className],
						split=split,
						sourcePath=imagePath.name,
					)
				)
	return samples


def resolveSamples(
	datasetDir: Path,
	manifestName: str,
	datasetSource: str,
) -> list[Sample]:
	if datasetSource == "folders":
		return loadSamplesFromFolders( datasetDir )
	if datasetSource == "manifest":
		return loadSamples( datasetDir, manifestName )

	cropsDir = datasetDir / "crops"
	if cropsDir.exists():
		folderSamples = loadSamplesFromFolders( datasetDir )
		if folderSamples:
			return folderSamples

	return loadSamples( datasetDir, manifestName )


def buildModel(pretrainedMode: str) -> tuple[nn.Module, str]:
	weights = None
	initialization = "random"

	if pretrainedMode in {"imagenet", "auto"}:
		try:
			weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
			initialization = "imagenet"
		except Exception:
			if pretrainedMode == "imagenet":
				raise

	model = mobilenet_v3_small( weights=weights )
	finalLayer = model.classifier[-1]
	if not isinstance( finalLayer, nn.Linear ):
		raise TypeError( "Unexpected MobileNetV3 classifier layout." )
	model.classifier[-1] = nn.Linear( finalLayer.in_features, len( CLASS_NAMES ) )
	return model, initialization


def buildSampler(samples: list[Sample]) -> WeightedRandomSampler:
	labelCounts = Counter( sample.label for sample in samples )
	weights = [1.0 / labelCounts[sample.label] for sample in samples]
	return WeightedRandomSampler(
		weights=torch.tensor( weights, dtype=torch.double ),
		num_samples=len( samples ),
		replacement=True,
	)


def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: str,
	lossFn: nn.Module,
) -> dict[str, float]:
	model.eval()
	totalLoss = 0.0
	totalCount = 0
	correct = 0
	truePositives = {className: 0 for className in CLASS_NAMES}
	falsePositives = {className: 0 for className in CLASS_NAMES}
	falseNegatives = {className: 0 for className in CLASS_NAMES}

	with torch.no_grad():
		for inputs, labels in loader:
			inputs = inputs.to( device )
			labels = labels.to( device )
			logits = model( inputs )
			loss = lossFn( logits, labels )
			totalLoss += float( loss.item() ) * labels.size( 0 )
			totalCount += labels.size( 0 )
			predictions = logits.argmax( dim=1 )
			correct += int( ( predictions == labels ).sum().item() )
			for classIndex, className in enumerate( CLASS_NAMES ):
				tp = int( ( ( predictions == classIndex ) & ( labels == classIndex ) ).sum().item() )
				fp = int( ( ( predictions == classIndex ) & ( labels != classIndex ) ).sum().item() )
				fn = int( ( ( predictions != classIndex ) & ( labels == classIndex ) ).sum().item() )
				truePositives[className] += tp
				falsePositives[className] += fp
				falseNegatives[className] += fn

	metrics: dict[str, float] = {
		"loss": totalLoss / max( totalCount, 1 ),
		"accuracy": correct / max( totalCount, 1 ),
	}
	for className in CLASS_NAMES:
		precisionDenom = truePositives[className] + falsePositives[className]
		recallDenom = truePositives[className] + falseNegatives[className]
		metrics[f"{className}_precision"] = (
			truePositives[className] / precisionDenom if precisionDenom else 0.0
		)
		metrics[f"{className}_recall"] = (
			truePositives[className] / recallDenom if recallDenom else 0.0
		)
		precision = metrics[f"{className}_precision"]
		recall = metrics[f"{className}_recall"]
		metrics[f"{className}_f1"] = (
			( 2.0 * precision * recall ) / ( precision + recall )
			if ( precision + recall )
			else 0.0
		)
	return metrics


def train(args: argparse.Namespace) -> int:
	repoRoot = Path( __file__ ).resolve().parent.parent
	datasetDir = ( repoRoot / args.dataset_dir ).resolve()
	outputDir = ( repoRoot / args.output_dir ).resolve()
	outputDir.mkdir( parents=True, exist_ok=True )

	setSeed( args.seed )
	device = resolveDevice( args.device )
	samples = resolveSamples( datasetDir, args.manifest, args.dataset_source )
	trainSamples = [sample for sample in samples if sample.split == "train"]
	valSamples = [sample for sample in samples if sample.split == "val"]
	if not trainSamples or not valSamples:
		raise ValueError( "Dataset manifest must contain both train and val samples." )

	trainDataset = IdentityDataset( trainSamples, args.image_size )
	valDataset = IdentityDataset( valSamples, args.image_size )
	trainLoader = DataLoader(
		trainDataset,
		batch_size=args.batch_size,
		sampler=buildSampler( trainSamples ),
		num_workers=args.workers,
	)
	valLoader = DataLoader(
		valDataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
	)

	model, modelInitialization = buildModel( args.pretrained )
	model = model.to( device )
	lossFn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	bestSelectionMetric = -1.0
	history: list[dict[str, float | int]] = []
	bestCheckpointPath = outputDir / "best.pt"
	metricsPath = outputDir / "metrics.json"

	for epoch in range( 1, args.epochs + 1 ):
		model.train()
		totalTrainLoss = 0.0
		totalTrainCount = 0
		trainCorrect = 0

		for inputs, labels in trainLoader:
			inputs = inputs.to( device )
			labels = labels.to( device )
			optimizer.zero_grad()
			logits = model( inputs )
			loss = lossFn( logits, labels )
			loss.backward()
			optimizer.step()

			totalTrainLoss += float( loss.item() ) * labels.size( 0 )
			totalTrainCount += labels.size( 0 )
			trainCorrect += int( ( logits.argmax( dim=1 ) == labels ).sum().item() )

		trainMetrics = {
			"loss": totalTrainLoss / max( totalTrainCount, 1 ),
			"accuracy": trainCorrect / max( totalTrainCount, 1 ),
		}
		valMetrics = evaluate( model, valLoader, device, lossFn )
		epochMetrics: dict[str, float | int] = {
			"epoch": epoch,
			"train_loss": trainMetrics["loss"],
			"train_accuracy": trainMetrics["accuracy"],
			"val_loss": valMetrics["loss"],
			"val_accuracy": valMetrics["accuracy"],
			"val_orange_precision": valMetrics["orange_precision"],
			"val_orange_recall": valMetrics["orange_recall"],
			"val_orange_f1": valMetrics["orange_f1"],
			"val_goblin_precision": valMetrics["goblin_precision"],
			"val_goblin_recall": valMetrics["goblin_recall"],
			"val_goblin_f1": valMetrics["goblin_f1"],
		}
		history.append( epochMetrics )
		print(
			f"epoch={epoch} "
			f"train_loss={trainMetrics['loss']:.4f} train_acc={trainMetrics['accuracy']:.4f} "
			f"val_loss={valMetrics['loss']:.4f} val_acc={valMetrics['accuracy']:.4f}"
		)

		selectionValue = float( epochMetrics[args.selection_metric] )
		if selectionValue > bestSelectionMetric:
			bestSelectionMetric = selectionValue
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"class_names": list( CLASS_NAMES ),
					"image_size": args.image_size,
					"model_initialization": modelInitialization,
					"selection_metric": args.selection_metric,
					"selection_metric_value": selectionValue,
				},
				bestCheckpointPath,
			)

	summary = {
		"device": device,
		"dataset_dir": str( datasetDir ),
		"train_samples": len( trainSamples ),
		"val_samples": len( valSamples ),
		"class_names": list( CLASS_NAMES ),
		"model_initialization": modelInitialization,
		"selection_metric": args.selection_metric,
		"history": history,
		"best_selection_metric": bestSelectionMetric,
		"best_checkpoint": str( bestCheckpointPath ),
	}
	metricsPath.write_text( json.dumps( summary, indent=2 ), encoding="utf-8" )
	print( f"Wrote checkpoint: {bestCheckpointPath}" )
	print( f"Wrote metrics: {metricsPath}" )
	return 0


def main() -> int:
	args = parseArgs()
	return train( args )


if __name__ == "__main__":
	raise SystemExit( main() )
