# loads and runs the nnet on the digit data
library('nnet')
X <- iris[,1:4]
y <- as.numeric(iris$Species)
yV <- class.ind(y)

nn<-nnet(X,yV, size=10, decay=1, MaxNWts=20000, maxit=100, rang=0.12)
yPred=predict(nn,X)

nCorrect = sum(max.col(yPred)==max.col(yV))
acc = nCorrect/nrow(yV)
print(acc)
