#Reviewer Experience Prediction

- System for predicting the number of game-hours that have been played by a reviewer given only the review text itself.

- Idea: The online gaming platform Steam publishes hundreds of thousands of video game reviews (along with other pieces of data) by its users and it also collects and makes available huge amounts of data linked to those very same users. So, we can look at things like...
    * How many hours a reviewer played the game he/she is reviewing
    * Whether a review was marked as helpful (and how many times this was done)
    * Whether a review was marked as humorous
    * Reviewer/user stats, such as:
        - Achievements attained
        - Number of hours played in the last 2 weeks

- Dependencies:
    - conda (anaconda/miniconda), which can be found [here](http://conda.pydata.org/miniconda.html)
        - For a Linux, 64-bit system, install with:
            ```curl http://repo.continuum.io/miniconda/Miniconda-1.6.0-Linux-x86_64.sh >! ~/Miniconda-1.6.0-Linux-x86_64.sh
            chmod a+x ~/Miniconda-1.6.0-Linux-x86_64.sh
            ~/Miniconda-1.6.0-Linux-x86_64.sh -b -p ~/conda
            rm -f ~/Miniconda-1.6.0-Linux-x86_64.sh```
        - MongoDB, which can be found [here](https://www.mongodb.org/downloads)

- Set-up:
    - Run util/setup.sh to create the conda environment (must have conda/miniconda installed, see `Dependencies` section above), which will also run Cython on the Cython modules so that they get compiled from source.
    - Activate the newly-created "reviews" environment: source activate reviews
    - Optional: If making changes to the Cython modules or to re-compile them, run util/cythonize.sh.
    - Optional: Start a MondoDB server by, for example, creating a screen session and running ```mongod```. To specify a path for the database, use the --dbpath option flag followed by the desired path. If in a screen session, press CTRL+a+d (not at the same time, but in a row) in order to "detach" from the session.
    - You're all set up! Now you will probably want to use make_train_test_sets.py to read the data in the "data" sub-directory into a MongoDB collection. Once the data is in the database, you can run the train.py and evaluate.py programs.