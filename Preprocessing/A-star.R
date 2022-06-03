install.packages("opencv","magick","image.textlinedetector")
library(opencv)
library(magick)
library(image.textlinedetector)

filelist = list.files(path =  "./DSS_binarized" ,pattern = ".jpg")
areas = NA
for (i in filelist) {
  base_path = getwd()
  path   <- paste(base_path,"/DSS_binarized/",i,sep = "")
  img    <- image_read(path)
  areas  <- image_textlines_astar(img, morph = TRUE, step = 2, mfactor = 5, trace = TRUE)
  area_overview = areas$overview
  write_path = paste(base_path,"/Outputs/combined/",i,"_combined_lines.jpg",sep = "")
  ocv_write(area_overview,write_path)
  
  for (j in 1:length(areas$textlines)) {
    cidr <- getwd()
    mkfldr <- "/Outputs/"
    dir.create(file.path(cidr,mkfldr,i))
    outpath = paste(getwd(),"/Outputs/",i,"/line",j,".jpg",sep = "")
    ocv_write(areas$textlines[[j]],outpath)
  }
}
