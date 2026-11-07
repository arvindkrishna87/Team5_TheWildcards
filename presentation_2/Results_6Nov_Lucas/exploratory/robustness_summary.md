# Robustness checks (exploratory)

Not part of the main results. Every run is included; none was chosen by test score.

Finished training jobs: 50 of 50. Failed or unfinished: 0.

## 1. Repeated cross-validation (4 split seeds, including 436)

 encoder setting  cases  accuracy  benign_recall  high_grade_recall
kimianet seed101   93.0     0.817          0.800              0.828
kimianet seed202   93.0     0.796          0.743              0.828
kimianet seed303   93.0     0.839          0.857              0.828
kimianet seed436   93.0     0.817          0.857              0.793
    uni2 seed101   93.0     0.892          0.857              0.914
    uni2 seed202   93.0     0.946          0.886              0.983
    uni2 seed303   93.0     0.935          0.914              0.948
    uni2 seed436   93.0     0.957          0.971              0.948

Cases where only one encoder was right, per split seed:

         only_kimianet  only_uni2
setting                          
seed101              7         14
seed202              2         16
seed303              1         10
seed436              1         14

## 2. Learning curve (seed 436; f = share of training cases)

 encoder setting  cases  accuracy  benign_recall  high_grade_recall
kimianet   f0.25   93.0     0.742          0.800              0.707
kimianet    f0.5   93.0     0.742          0.857              0.672
kimianet    f1.0   93.0     0.817          0.857              0.793
    uni2   f0.25   93.0     0.849          0.914              0.810
    uni2    f0.5   93.0     0.839          0.914              0.793
    uni2    f1.0   93.0     0.957          0.971              0.948

## 3. Logistic regression baseline (no attention) vs attention model, seed 436

Baseline:

 encoder  cases  accuracy  benign_recall  high_grade_recall
kimianet   93.0     0.828          0.686              0.914
    uni2   93.0     0.903          0.914              0.897

Attention model:

 encoder  cases  accuracy  benign_recall  high_grade_recall
kimianet   93.0     0.817          0.857              0.793
    uni2   93.0     0.957          0.971              0.948

## 4. Stain knockout (no retraining; cases missing the kept stains are left out)

Check: the 'all' row matches the main predictions to within 0.000000.

 encoder     stains  cases  accuracy  benign_recall  high_grade_recall
kimianet        all   93.0     0.817          0.857              0.793
kimianet   h&e only   93.0     0.645          0.657              0.638
kimianet melan only   91.0     0.846          0.800              0.875
kimianet     no h&e   92.0     0.815          0.857              0.789
kimianet   no melan   93.0     0.591          0.943              0.379
kimianet   no sox10   93.0     0.839          0.800              0.862
kimianet sox10 only   91.0     0.462          0.941              0.175
    uni2        all   93.0     0.957          0.971              0.948
    uni2   h&e only   93.0     0.871          0.714              0.966
    uni2 melan only   91.0     0.703          0.971              0.536
    uni2     no h&e   92.0     0.685          0.971              0.509
    uni2   no melan   93.0     0.946          0.943              0.948
    uni2   no sox10   93.0     0.946          0.943              0.948
    uni2 sox10 only   91.0     0.571          0.971              0.333
