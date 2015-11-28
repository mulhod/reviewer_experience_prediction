# Reviewer Experience Prediction

## Original Inspiration
- Create a system for predicting the number of game-hours that have been played by a reviewer given only the review text itself.
- Idea: The online gaming platform Steam publishes hundreds of thousands of video game reviews (along with other pieces of data) by its users and it also collects and makes available huge amounts of data linked to those very same users. So, we can look at things like...
    * How many hours a reviewer played the game he/she is reviewing
    * Whether a review was marked as helpful (and how many times this was done)
    * Whether a review was marked as humorous
    * Reviewer/user stats, such as:
        - Achievements attained
        - Number of hours played in the last 2 weeks

## Current -- and More General -- Aim
- Use the system to conduct machine learning experiments of various kinds, not just in trying to predict number of game-hours played.

## Dependencies
- Conda (anaconda/miniconda), which can be found [here](http://conda.pydata.org/miniconda.html)
    * For a Linux, 64-bit system, install with:
      
```
         wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
         chmod a+x Miniconda-latest-Linux-x86_64.sh
         ./Miniconda-latest-Linux-x86_64.sh -b -p conda
         rm -f Miniconda-latest-Linux-x86_64.sh
```
      
- MongoDB, which can be found [here](https://www.mongodb.org/downloads)

## Set-up
- Run ```util/setup.sh``` to create the conda environment (must have conda installed, see `Dependencies` section above), which will also run setup.py on the Cython modules so that they get compiled from source.
- Activate the newly-created "reviews" environment: ```source activate reviews```
- Optional: If making changes to the Cython modules or to re-compile them, run ```python util/setup.py build_ext``` (or ```util/cythonize.sh``` if you want to run the commands directly). If a "build" directory already exists in the root of the repository clone, setup.py sometimes does not work correctly, even with the --inplace flag. Simply deletng the "build" directory should solve the problem.
- Optional: Start a MondoDB server by, for example, creating a screen session and running ```mongod```. To specify a path for the database, use the ```--dbpath``` option flag followed by the desired path. If in a screen session, press CTRL+a+d (not at the same time, but in a row) in order to "detach" from the session.
- You're all set up!

## Use
- Distributed with this package is pre-collected data for just over 10 games (in the `data` directory), so there is no need to actually run the web scraping utility (```get_review_data```).
- You will probably want to first use the ```make_train_test_sets``` utility to read the data in the ```data``` sub-directory into a MongoDB collection. You must have a MongoDB server running for this to work, obviously.
- Once the data is in the database, you can run the command-line utilties. Use ```extract_features``` to get the NLP features for each review and add them to the MongoDB database). If NLP features are extracted for all of the data available, the database will currently balloon to about 230GiB, so it is very important to try to limit the feature extraction if resources are tight. Once features are in the database, learning experiments can be conducted. Use ```learn``` for running incremental machine learning experiments. A number of learning algorithms can be used as well as default parameter grids so that each experiment will try every possibility and output the results of the model evaluations at each step of the learning process.
