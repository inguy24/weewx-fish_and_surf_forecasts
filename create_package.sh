#!/bin/bash
#
# WeeWX Surf & Fishing Forecast Extension Package Creator
# Creates installation zip package for WeeWX extension system
# Phase II: Local Surf & Fishing Forecast System
# Magic Animal: Seahorse 🐟
#

set -e

VERSION="1.0.0-alpha"
EXTENSION_NAME="weewx-surf-fishing"
PACKAGE_NAME="${EXTENSION_NAME}-${VERSION}"
BUILD_DIR="build"
PACKAGE_DIR="${BUILD_DIR}/${EXTENSION_NAME}"

echo "Creating WeeWX Surf & Fishing Forecast Extension Package v${VERSION}"
echo "Magic Animal: Seahorse 🐟"
echo "=========================================================================="

# Clean and create build directory
echo "Setting up build directory..."
rm -rf ${BUILD_DIR}
mkdir -p ${PACKAGE_DIR}/bin/user

# Copy required files
echo "Copying extension files..."

# Core files
echo "  Copying install.py..."
cp install.py ${PACKAGE_DIR}/ || { echo "Error: install.py not found"; exit 1; }

echo "  Copying MANIFEST..."
cp MANIFEST ${PACKAGE_DIR}/ || { echo "Error: MANIFEST not found"; exit 1; }

# Service implementation
echo "  Copying bin/user/surf_fishing.py..."
cp bin/user/surf_fishing.py ${PACKAGE_DIR}/bin/user/ || { echo "Error: bin/user/surf_fishing.py not found"; exit 1; }

echo "  Copying bin/user/surf_fishing_fields.yaml..."
cp bin/user/surf_fishing_fields.yaml ${PACKAGE_DIR}/bin/user/ || { echo "Error: bin/user/surf_fishing_fields.yaml not found"; exit 1; }

# Documentation
echo "  Copying README.md..."
cp README.md ${PACKAGE_DIR}/ || { echo "Error: README.md not found"; exit 1; }

echo "  Copying CHANGELOG.md..."
cp CHANGELOG.md ${PACKAGE_DIR}/ || { echo "Error: CHANGELOG.md not found"; exit 1; }

# Verify all files are present
echo "Verifying package contents..."
echo "Required files:"

# Read from the copied MANIFEST file in the package directory and debug each line
while IFS= read -r line; do
    # Remove any carriage returns and leading/trailing whitespace
    file=$(echo "$line" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Skip comments (lines starting with #) and empty lines
    if [[ "$file" =~ ^#.*$ ]] || [[ -z "$file" ]]; then
        echo "  Skipping: '$file'"
        continue
    fi
    
    # Check if file exists
    if [ -f "${PACKAGE_DIR}/${file}" ]; then
        echo "  ✓ ${file}"
    else
        echo "  ✗ MISSING: ${file}"
        echo "    Expected path: ${PACKAGE_DIR}/${file}"
        echo "    Actual directory contents:"
        ls -la "${PACKAGE_DIR}/"
        if [ -d "${PACKAGE_DIR}/bin/user" ]; then
            echo "    bin/user directory contents:"
            ls -la "${PACKAGE_DIR}/bin/user/"
        fi
        exit 1
    fi
done < "${PACKAGE_DIR}/MANIFEST"

# Create the zip package
echo ""
echo "Creating installation package..."
cd ${BUILD_DIR}
zip -r ../${PACKAGE_NAME}.zip ${EXTENSION_NAME}/

# Move back and show results
cd ..
PACKAGE_SIZE=$(du -h ${PACKAGE_NAME}.zip | cut -f1)

echo ""
echo "Package created successfully!"
echo "=========================="
echo "Package: ${PACKAGE_NAME}.zip"
echo "Size: ${PACKAGE_SIZE}"
echo "Magic Animal: Seahorse 🐟"
echo ""

# Show package contents
echo "Package contents:"
unzip -l ${PACKAGE_NAME}.zip

echo ""
echo "Installation commands:"
echo "  weectl extension install ${PACKAGE_NAME}.zip"
echo "  sudo systemctl restart weewx"
echo ""
echo "Prerequisites:"
echo "  - WeeWX 5.1+ must be installed"
echo "  - Phase I Marine Data Extension must be installed first"
echo "  - Internet connection required for GRIB processing libraries"
echo ""
echo "The package is ready for distribution!"
echo "Phase II: Local Surf & Fishing Forecast System"