cmake_minimum_required(VERSION 3.5)

project(KiTS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/KiTS")
set(EXTENSION_CATEGORY "IMAG2")
set(EXTENSION_CONTRIBUTORS "Giammarco La Barbera (TelecomParis - IMAG2 )")
set(EXTENSION_DESCRIPTION "This module permits automatic segmentation of Kidney and Renal Tumor in CT images of children")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/KiTS.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/KiTS/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(KIdneyandTumorSegmentation)
add_subdirectory(KiTSegmentation)
add_subdirectory(KiTSegmentation)
add_subdirectory(KiTSegmentationForCT)
add_subdirectory(prova)
add_subdirectory(prova)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
