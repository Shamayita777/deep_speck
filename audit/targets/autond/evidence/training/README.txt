AutoND fixed-difference reproduction evidence
Source commit: 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779
Mean of five test accuracies: 0.5097414
Population SD: 0.0005999
Sample SD: 0.0006707
Published Table 6 comparison target: 0.5096

Checkpoint selection uses val_loss. The round's reported best
validation accuracy is not necessarily the saved checkpoint's
own validation accuracy. See checkpoint_epoch_diagnostic.csv.

This run does not reproduce AutoND's complete evolutionary
input-difference search. Download the Kaggle notebook separately
to preserve the full cell output and training logs.
