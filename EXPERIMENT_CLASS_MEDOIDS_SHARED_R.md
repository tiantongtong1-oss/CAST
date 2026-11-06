# Class medoids with one class-balanced shared Q90 radius

Parent: experiment/fused-knn-l2-radius at 94f7e91d91ce38e02e334a3d42861c9b25a568a7.
Each valid source true-label class with >=2 nonzero finite features gets an exact medoid in L2-normalized feature space. Every source row measures distance to its OWN class medoid. Give each ready class equal mass (row weights 1/n_c, normalized across ready classes) and select the first sorted distance with cumulative mass >=0.90. Weighted empirical inverse-CDF, no interpolation; not an average of class Q90s. There is ONE scalar shared R, not seven radii.

Target original-view normalized feature checks distance to the medoid selected by its TEACHER PSEUDO-LABEL. Strict d<R, no 0.7 multiplier. Missing/singleton classes cannot radius-rescue. Centers/counts/readiness/R are rebuilt every epoch. KNN still uses all valid source/target memory, not per-class-filtered neighbors. The prototype-loss angular centers remain separate from radius medoids.

Final selection: agreement AND (confidence OR enabled class-center radius) AND unchanged equal-weight source-label/target-probability kNN fusion; default fusion support .7, radius warmup 3. Lambda, threshold EMA and prototype loss unchanged.

Parent runtime typos torch(conf1,conf2) and doubsle are corrected to cat BOTH view confidence arrays and double(), matching the user's successful local baseline. Mean statistics over 2N views; logs renamed TwoView/View_Count. Model suffix classmedoid_sharedr preserves previous outputs. Historical parent experiment documents are superseded by this file.

Validation: syntax/static integration checks; added tensor tests for class-balanced quantile, class-dependent query centers, scaling, class duplication, invalid labels, singleton/missing/empty memory and strict zero radius. PyTorch/GPU execution unavailable locally.
