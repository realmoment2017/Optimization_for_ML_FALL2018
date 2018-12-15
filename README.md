# Optimization for Machine Learning FALL2018

In  this  project,  we  explore  different  optimizationalgorithms   for   binary   classification   tasks.   Since   SVMs   is   a widely  used  algorithm  for  binary  classification,  we  choose  L1-SVM  to  be  our  objective  function.  We  implemented  Pegasos,Primal   Cyclic   Coordinate   Descent(CCD)   Primal   RandamizedCoordinate  Descent(RCD)  and  two  versions  of  Sequential  Min-imizing  Optimization(SMO/RSMO).  We  compare  performancesof  different  algorithms  with  respect  to  the  number  of  iterationsand   the   running   time.   We   also   studied   the   effects   of   otherhyperparameters,  such  as  the  choices  ofλs,  choices  of  slackpenaltyCs and the size of the dataset. All algorithms are trained and  tested  on  MNIST  and  Cifar10,  which  are  two  of  the  most popular datasets in academia.
### Team Member

  - Ji Han
  - Jiteng Mu
  - Xin Ren
  - Yikuang Yang

### References

* [LasseRegin](https://github.com/LasseRegin/SVM-w-SMO)
* [Stanford](http://cs229.stanford.edu/materials/smo.pdf)
* [Jon](https://jonchar.net/notebooks/SVM/)
