mkdir /root/faiss
mkdir /root/faiss/build
ln -s /gayu-west/faiss ~/faiss/build/faiss
cp -r /root/faiss/build/faiss/python /root/ 
cd /root/python
python setup.py install

python -m pip uninstall -y numpy
python -m pip uninstall -y numpy
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"