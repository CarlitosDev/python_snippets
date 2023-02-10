'''

microsoft_recommenders.py



source ~/.bash_profile && python3 -m pip install recommenders


cd "/Users/carlos.aguilar/Documents/DS_repos"
mkdir microsoft_recommenders
git clone https://github.com/microsoft/recommenders microsoft_recommenders
cd microsoft_recommenders
source ~/.bash_profile && python3 setup.py install


>> 	AttributeError: 'dict' object has no attribute '__LIGHTFM_SETUP__'

'''


- Lumiere: 
	* Upload all the analysed videos into ES.
	* Work out an alternative to the cosine distance issue (not native in AWS ES).
	* (Once ES works) Re-implement front-end to call ES instead of local files.

- PRISM:
	* Cluster migration planned for Tuesday the 2nd. I have been ask to be on-call and carry quality tests post-switch.

- Learner profile:
	* Sketch a possible solution for classifiying students. Include variable weights and an informative approach (ie: next steps to jump levels).

- Student feedback:
	* Show Tim the QA approach and possible directions.

- CPF:
	* Change some the views to incorporate some of the information requested by the users.