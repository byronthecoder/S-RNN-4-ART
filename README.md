# S-RNN-4-ART
 A neural network designed to detect and measure phonetic convergence and speech imitation.


This code repository is a demo of the Siamese RNN model described in the [paper](https://doi.org/10.48550/arXiv.2306.05088):


    "The ART of Conversation: Measuring Phonetic Convergence and Deliberate Imitation in L2-Speech with a Siamese RNN"
    
    Zheng Yuan, Aldo Pastore, Dorina de Jong, Hao Xu, Luciano Fadiga and Alessandro D’Ausilio

    (preceding the publication InterSpeech 2023)


The folder structure is as follows:

    - data: 
        - FR: the French dataset, already built with the scripts in the folder ./src.
            - mfcc_dd: the 39-dimensional MFCCs of the French data.
            - tfrecords: the data examples for training and testing the model.
            - wav: 32 demos of the L2 English speech from French speakers (id: 42, 43)

        - ITA: the Italian dataset, only containing the wav files. For the user to test the scripts by themselves.
            - mfcc_dd: no files yet.
            - tfrecords: no files yet.
            - wav: 32 demos of the L2 English speech from Italian speakers (id: 1, 2)
    
    - models:
        - FR: the best model trained solely on the French dataset.
        - ITA: the best model trained solely on the Italian dataset.
        - FR_ITA: the best model trained on the French and Italian datasets.

    - results: A folder for saving the results of experiments. Contains one test demo.

    - src: Some scripts have flags to help the user configure the script. Use the flag -h to see the usage.

        - Config.py: stores the paths to the data, models, results, and some global variables.
        - MFCC_extraction.py: extracts the MFCCs from the wav files.
            use flag -i/-f to specify the dataset.
        - Create_tfrecords.py: creates the tfrecords files from the MFCCs.
            use flag -i/-f to specify the dataset.
        - Siamese_models.py: the model architecture, containing 4 different architectures.
        - Train_Siamese_RNN.py: the training script.
            use flag -i/-f/-a to specify the dataset.
        - Test.Siamese_RNN.py: the testing script.
            e.g. -i -I -v -c (test with the ITA model on the ITA dataset; do the validation; select the imitation data)
        - Utilities.py: some useful functions.
       
    - README.txt: a brief user guide.

    - SiameseRNN_env.yml: a configuration file for the conda environment.
    - Requirements.txt: a list of the dependencies of the project. 


The authors have tested the scripts to ensure that they work as expected. 
However,if the user runs into some bugs, please bear with us.

## License

This project is licensed under the CC BY-NC-ND 4.0 License - see the [LICENSE](LICENSE) file for details.

[![License:  CC BY-NC-ND 4.0](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
