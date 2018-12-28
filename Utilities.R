#Read train images
read_image_untouched = function(filename, no_samples) {
  to.read<-file(filename, "rb") #"train-images.idx3-ubyte"
  readBin(to.read, integer(), n=4, endian="big") #skip header
  trimg<-array(NA, c(no_samples, 28*28))
  for(i in 1:no_samples) {
    trimg[i,]<-readBin(to.read,integer(), size=1, n=28*28, endian="big", signed=FALSE);
  }
  #thresholding
  #trimg[,][trimg[,] >= 128]<-255
  #trimg[,][trimg[,] < 128]<-0
  
  return(trimg)
}

read_label = function(filename, no_samples) {
  #Read train labels
  to.read<-file(filename, "rb") #"train-labels.idx1-ubyte"
  readBin(to.read, integer(), n=2, endian="big") #skip header
  trlbl<-readBin(to.read,integer(), size=1, n=no_samples, endian="big", signed=FALSE)
  
  return(factor(trlbl))
}

# reference https://stackoverflow.com/questions/10865489/scaling-an-r-image
resizePixels = function(im, w, h) {
  pixels = as.vector(im)
  # initial width/height
  w1 = nrow(im)
  h1 = ncol(im)
  # target width/height
  w2 = w
  h2 = h
  # Create empty vector
  temp = vector('numeric', w2*h2)
  # Compute ratios
  x_ratio = w1/w2
  y_ratio = h1/h2
  # Do resizing
  for (i in 1:h2) {
    for (j in 1:w2) {
      px = floor(j*x_ratio)
      py = floor(i*y_ratio)
      if (py==0 && px==0) {
        px = 1
      }
      temp[(i*w2)+j] = pixels[(py*w1)+px]
    }
  }
  m = matrix(temp, h2, w2)
  return(m)
}

bound_image = function(img_untouched) {
  no_samples = nrow(img_untouched)
  trimg_res<-array(NA, c(no_samples, 20*20))
  for (i in 1:no_samples) {
    x<-matrix(img_untouched[i,], 28, 28)
    x<-x[which(rowSums(x) > 0),]
    x<-x[,which(colSums(x) > 0)]
    trimg_res[i,]<-resizePixels(x,20,20)
  }
  return(trimg_res)
}