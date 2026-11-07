# Same-input comparison (exploratory, not part of the main Presentation 2 results)

Question: what happens if KimiaNet and UNI2 receive the same input?

Main results (presentation2/): KimiaNet uses each patch at its original size; UNI2 uses patches resized to 224x224.

Test A: KimiaNet also uses patches resized to 224x224, compared with the main UNI2 run.
Test B: UNI2 also uses patches at their original size (edges padded to a multiple of 14), compared with the main KimiaNet run.

Same 5 split files, seed 436 and training settings as the main runs.
Nothing in presentation2/ is changed. Results are in results_summary/ in this folder.
