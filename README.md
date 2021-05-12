# Automated-Driver-Seating-and-Alert-System-Using-Supervised-Machine-Learning
A system providing vehicle interior adjustments (here driver seat position) as per user preferences by automatically identifying the user, using face recognition.

Description:
Vehicles these days are increasingly smart and offer quite a few features to assist
the driver or to make the user experience seamless. Over the past few years lots
of automation has been introduced in the automotive industry with the help of
advances in machine learning discipline. In this project we are enhancing the
vehicle user experience and safety by introducing a machine learning aspect to
the driver seating adjustments and assessing user’s expression with the help of
computer vision techniques, respectively. This project presents an approach to save
the user’s seating preference and integrate it with their facial identity and to alert
the driver with chime alert if they are deemed to be drowsy based on their facial
expression. Local Binary Patterns Histograms is used as the core model for facial
recognition. The project offers a demo that shows how the seating positions are
adjusted based on the recognized face of an established user, the ability to add
new user data or update the preferred seating position for an established user. For
the visualization of seating position adjustments, we have used pycairo graphics
library.

1. Create a folder named dataset and another one named trainer inside the folder with your code (in this case inside Driver Seat folder)
2. Install shape_predictor_68_face_landmarks.dat file
3. Run Projallin.py file

The flow works as follows,

![Flow](https://user-images.githubusercontent.com/83297868/117921800-5fc25100-b2bf-11eb-872a-1d41150020e9.png)

The different seating arrangements provided here are as follows,
![Seatsall](https://user-images.githubusercontent.com/83297868/117921969-aa43cd80-b2bf-11eb-8e1e-a6156b8681a1.png)
