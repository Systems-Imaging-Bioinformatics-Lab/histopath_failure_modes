To run img_manip from the command line:
`python img_manip.py inputFileName type [outputFileName]`
1. `inputFileName` is the file you want to run the artifact addition on
2. `type` should be one of the members of the list ['bubbles','marker','fold','sectioning','illumination','tear','stain']
3. `outputFileName` is optional, and can be specified explicitly.  
  Otherwise, the output is saved at pwd + inputFileName + _ + type[0:4] + .jpeg
