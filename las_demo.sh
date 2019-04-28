#!/bin/bash
# Need to replace with your own directory path.
# Only the mac user will need the WINE_DIR
WINE_DIR="/usr/local/bin//wine"
LAS_TOOL_DIR="/Users/ethanchen/LAStools/bin"

LAS_GROUND="${LAS_TOOL_DIR}/lasground.exe"
LAS_HEIGHT="${LAS_TOOL_DIR}/lasheight.exe"
LAS_CLASSIFY="${LAS_TOOL_DIR}/lasclassify.exe"

TARGET_LAS="SP01.las"
GROUND_OUTPUT=${TARGET_LAS/".las"/"._g.las"}
HEIGHT_OUTPUT=${TARGET_LAS/".las"/"._gh.las"}
CLASS_OUTPUT=${TARGET_LAS/".las"/"._ghc.las"}


# For mac
${WINE_DIR} ${LAS_GROUND} -i "${TARGET_LAS}" -o "${GROUND_OUTPUT}" -step 1 -not_airborne

${WINE_DIR} ${LAS_HEIGHT} -i ${GROUND_OUTPUT} -o "${HEIGHT_OUTPUT}" 

${WINE_DIR} ${LAS_CLASSIFY} -i ${HEIGHT_OUTPUT} -o "${CLASS_OUTPUT}"


# For Window commit up the 'For mac' part and uncomment below.

#${LAS_GROUND} -i "${TARGET_LAS}" -o "${GROUND_OUTPUT}" -step 1

#${LAS_HEIGHT} -i ${GROUND_OUTPUT} -o "${HEIGHT_OUTPUT}" 

#${LAS_CLASSIFY} -i ${HEIGHT_OUTPUT} -o "${CLASS_OUTPUT}"
