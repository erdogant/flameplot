echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf flameplot.egg-info

echo "Making new wheel.."
echo ""
python setup.py bdist_wheel
echo ""

echo "Making source build .."
echo ""
python setup.py sdist
echo ""

read -p "Press [Enter] to install the pip package..."
pip install -U dist/flameplot-0.1.2-py3-none-any.whl
echo ""

read -p ">twine upload dist/* TO UPLOAD TO PYPI..."
echo ""

read -p "Press [Enter] key to close window..."
