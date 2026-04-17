import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use( "Agg" )

import matplotlib.pyplot as plt


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot identity-classifier training metrics from metrics.json."
	)
	parser.add_argument(
		"--metrics",
		default="artifacts/identity_classifier/metrics.json",
		help="Path to metrics.json produced by train_identity_classifier.py.",
	)
	parser.add_argument(
		"--output",
		default="artifacts/identity_classifier/metrics.png",
		help="Output image path.",
	)
	return parser.parse_args()


def loadMetrics(metricsPath: Path) -> dict:
	return json.loads( metricsPath.read_text( encoding="utf-8" ) )


def main() -> int:
	args = parseArgs()
	repoRoot = Path( __file__ ).resolve().parent.parent
	metricsPath = ( repoRoot / args.metrics ).resolve()
	outputPath = ( repoRoot / args.output ).resolve()
	outputPath.parent.mkdir( parents=True, exist_ok=True )

	metrics = loadMetrics( metricsPath )
	history = metrics.get( "history", [] )
	if not history:
		raise ValueError( f"No history entries found in {metricsPath}" )

	epochs = [entry["epoch"] for entry in history]
	trainLoss = [entry["train_loss"] for entry in history]
	valLoss = [entry["val_loss"] for entry in history]
	trainAccuracy = [entry["train_accuracy"] for entry in history]
	valAccuracy = [entry["val_accuracy"] for entry in history]
	valOrangePrecision = [entry["val_orange_precision"] for entry in history]
	valOrangeRecall = [entry["val_orange_recall"] for entry in history]
	valGoblinPrecision = [entry["val_goblin_precision"] for entry in history]
	valGoblinRecall = [entry["val_goblin_recall"] for entry in history]

	plt.style.use( "seaborn-v0_8-whitegrid" )
	figure, axes = plt.subplots( 2, 2, figsize=( 12, 8 ) )
	figure.suptitle( "Identity Classifier Training Metrics", fontsize=14, fontweight="bold" )

	lossAxis = axes[0][0]
	lossAxis.plot( epochs, trainLoss, marker="o", label="train_loss", color="#d97706" )
	lossAxis.plot( epochs, valLoss, marker="o", label="val_loss", color="#1d4ed8" )
	lossAxis.set_title( "Loss" )
	lossAxis.set_xlabel( "Epoch" )
	lossAxis.set_ylabel( "Loss" )
	lossAxis.legend()

	accuracyAxis = axes[0][1]
	accuracyAxis.plot( epochs, trainAccuracy, marker="o", label="train_accuracy", color="#059669" )
	accuracyAxis.plot( epochs, valAccuracy, marker="o", label="val_accuracy", color="#7c3aed" )
	accuracyAxis.set_title( "Accuracy" )
	accuracyAxis.set_xlabel( "Epoch" )
	accuracyAxis.set_ylabel( "Accuracy" )
	accuracyAxis.set_ylim( 0.0, 1.05 )
	accuracyAxis.legend()

	orangeAxis = axes[1][0]
	orangeAxis.plot( epochs, valOrangePrecision, marker="o", label="orange_precision", color="#ea580c" )
	orangeAxis.plot( epochs, valOrangeRecall, marker="o", label="orange_recall", color="#fb7185" )
	orangeAxis.set_title( "Orange Metrics" )
	orangeAxis.set_xlabel( "Epoch" )
	orangeAxis.set_ylabel( "Score" )
	orangeAxis.set_ylim( 0.0, 1.05 )
	orangeAxis.legend()

	goblinAxis = axes[1][1]
	goblinAxis.plot( epochs, valGoblinPrecision, marker="o", label="goblin_precision", color="#0f766e" )
	goblinAxis.plot( epochs, valGoblinRecall, marker="o", label="goblin_recall", color="#2563eb" )
	goblinAxis.set_title( "Goblin Metrics" )
	goblinAxis.set_xlabel( "Epoch" )
	goblinAxis.set_ylabel( "Score" )
	goblinAxis.set_ylim( 0.0, 1.05 )
	goblinAxis.legend()

	bestEpoch = max( history, key=lambda entry: entry["val_accuracy"] )["epoch"]
	for axis in axes.flat:
		axis.axvline( bestEpoch, color="#111827", linestyle="--", linewidth=1, alpha=0.45 )

	bestAccuracy = metrics.get( "best_val_accuracy", 0.0 )
	figure.text(
		0.5,
		0.01,
		f"best_epoch={bestEpoch}  best_val_accuracy={bestAccuracy:.4f}  "
		f"train_samples={metrics.get('train_samples', 0)}  val_samples={metrics.get('val_samples', 0)}",
		ha="center",
		fontsize=10,
	)

	plt.tight_layout( rect=( 0, 0.04, 1, 0.96 ) )
	figure.savefig( outputPath, dpi=160 )
	plt.close( figure )
	print( outputPath )
	return 0


if __name__ == "__main__":
	raise SystemExit( main() )
