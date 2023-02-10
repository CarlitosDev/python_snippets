'''
quantum_NLP.py

pip3 install cython numpy depccg
pip3 install lambeq[depccg]

#depccg_en download
# for me this one works
python3 -m depccg en download


# I am using this installation instead
bash <(curl 'https://cqcl.github.io/lambeq/install.sh')


At a high level, the library allows the conversion of any sentence to a quantum circuit, based on a given compositional model and certain parameterisation and choices of ansätze.


'''


from lambeq.ccg2discocat import DepCCGParser

depccg_parser = DepCCGParser()
diagram = depccg_parser.sentence2diagram('This is a test sentence')
diagram.draw()



diagram = depccg_parser.sentence2diagram('This is extracted from a video that I am about to parse')
diagram.draw()