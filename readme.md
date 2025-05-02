# Face Identification

<img src="pics\result_image.jpg" width="1000">

Face identification is a technology that uses algorithms and computer vision to analyze facial features in an image and compare them to a database of known faces to determine a match. It's a form of biometric identification that verifies or identifies a person based on their facial features. 

<img src="pics\result_image_0.jpg" width="1000">

This deep learning project belongs to voice classification of 22 people. This project is built with the aid of [pydub](https://github.com/jiaaro/pydub). Finally, a telegram bot has been created that receives speech and predicts the name of the speaker.


## How to install
Run this command:
```
pip install -r requirements.txt
```

## How to run
+ For each person whose face image you want to have identified, create a folder in `face_bank` directory.
+ Put a few images of each person in the folder corresponding to that person. The photos in this folder should contain only a single face image.
+ Put the image, in which you want to identify faces, in `input` directory.
+ By running the following command, the file `face_bank.npy` is created and the faces are then identified.

```
python face_identification_obj_orntd.py --update
```
+ You can see the result in `output` directory.

Note: You don't need to create `face_bank.npy` every time. After `face_bank.npy` is created, you can identify people faces by running the following command:

```
python face_identification_obj_orntd.py --image PATH/TO/IMAGE/FILE.
```

### What if we want to identify a new person's image?

To identify a new person's image, create a folder for every new person in `face_bank` directory and put there a few images of that person in it. If you no longer want a person's face to be identified, you can delete their folder from the `face_bank` folder. Again, update the `face_bank.npy` by running the following command and enjoy it.

```
python face_identification_obj_orntd.py --update
```

The following images show the results before and after updating the face_bank.

