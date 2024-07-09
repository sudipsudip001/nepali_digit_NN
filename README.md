# nepali_digit_NN
this-> Neural Network can learn and classify 'Nepali' handwritten digits (क-ज्ञ 0-९) made using numpy only. It's a scratch implementation of Neural Network and gives an intuition of how things really work.

### I'd like to thank [Prasanna1991](https://github.com/Prasanna1991/DHCD_Dataset) for the dataset.

To talk about the Neural Network architecture, it's a pretty simple one in terms of implementation. There're `46` classes(`36` Nepali alphabets and `10` digits) resulting in 46 neuron output layer. The input is of pixel size `32*32` resulting in `1024` input neurons. I've just arbitrarily chosen `46*1` neurons hidden layer for now. Talking about the model's accuracy, though it achieves a considerable `92%` accuracy for training set and `86%` for testing set, it's not enough considering the state of the art. Although we can consider it a very good start given that it doesn't contain anything fancy. Just an input, output and a hidden layer. Still, a lot needs to be done in order to achieve the benchmarks that has previously been achieved in this dataset. I've played around with the `learning rate` and `iterations`. '1' seems to be the better choice for learning rate and accuracy pretty much goes up with the increase in iterations. 'Softmax' has been used in the output layer for multi-class classification with 'ReLU' in the hidden layer so I don't think there's much to tinker with that. Adding 1-2 more hidden layers might work which I shall try.

To run the program:
1. Create a folder and open a terminal as administrator in it.
2. Clone the github repository:
`git clone https://github.com/sudipsudip001/nepali_digit_NN.git`
3. Go into the 'nepali_digit_NN' folder using the command:
`cd nepali_digit_NN`
4. Open terminal at this folder and paste the following command:
`pip install -r requirements.txt`
5. Start the notebook and explore!
`jupyter-notebook`

To run the flask application:
1. Follow till step 3.
2. Create a virtual environment for preventing any conflicts: <br>
   `python -m venv .venv` <br>
   `source .venv/bin/activate`
3. Install the dependencies:
   `pip install -r requirements.txt`
4. Run the flask app:
   `flask --app app.py run`
5. The server will run on `(http://127.0.0.1:5000)`
6. Insert the image and check the prediction.

(Note 9th July, 2024: The predictions are poor and I still have a lot to work for proper preprocessing in order to improve the accuracy!)
