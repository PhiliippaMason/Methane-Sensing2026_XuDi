# convert_prisma.R

# load library
# if (!requireNamespace("prismaread", quietly = TRUE)) {
#   remotes::install_github("lbusett/prismaread")
# }
library(prismaread)
library(hdf5r)

# Parameter settings
in_file    = "./PRISMA/method_valid/PRS_L1_STD_OFFL_20210209072220_20210209072224_0001.he5"
out_folder = "./PRISMA/method_valid/"
out_format = "ENVI"

# Output directory
# dir.create(out_folder, recursive = TRUE, showWarnings = FALSE)

# Conversion
pr_convert(in_file    = "./PRISMA/method_valid/PRS_L1_STD_OFFL_20210209072220_20210209072224_0001/PRS_L1_STD_OFFL_20210209072220_20210209072224_0001.he5",
           out_folder = "./PRISMA/method_valid/PRS_L1_STD_OFFL_20210209072220_20210209072224_0001/",
           out_format = "ENVI",
           VNIR        =  FALSE , 
           SWIR        =  TRUE , 
           FULL        =  FALSE , 
           LATLON      =  FALSE , 
        #    PAN         =  TRUE , 
           overwrite   =  TRUE,
           CLOUD  = TRUE
           )

cat("Conversion completed\n")
