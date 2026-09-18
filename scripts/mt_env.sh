# Source from bash: puts the project and its pip --target environment on PYTHONPATH.
ROOT=/home/morg/NLP_2526b/tomshabtay/tau_nlp_project
export PYTHONPATH=$ROOT/.mt_env/lib:$ROOT
export PATH=$ROOT/.mt_env/lib/bin:$PATH
cd $ROOT
# Python headers for Triton's runtime C helper (compiled at first use); some GPU nodes (e.g. n-804) lack python3-dev
export CPATH=$ROOT/.mt_env/include/python3.12${CPATH:+:$CPATH}
