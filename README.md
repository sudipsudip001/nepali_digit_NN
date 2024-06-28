# nepali_digit_NN
Here's an equivalent code of the 'main' code in pytorch. I simply did it to increase the compute i.e. to increase the number of trainings I do in neural network. Harnessing the power of 'cuda' makes the computing so so fast. If you've been training on CPU and switch to GPU you'd have definitely felt the joy of how fast things actually worked. For a moment perhaps I felt what Ian and Andrew or even Illya, Alex and Geoffrey felt in their quest to transfer the compute to cuda.

There's simply been a slight modification in terms of actual code. I've added learning rate optimizer in the program so the training wouldn't plateu. A circular learning rate optimization has been used. Though it has no effect in terms of our training set accuracy but it significantly increases our training set accuracy. I tried with other learning rate optimizers like StepLR, ReduceLRonPlateu, but this seemed to work best. Adaptive learning rate simply helps prevent the training set from being stuck. The major talking point here is that it still isn't showing no improvement in training set and it bothers me.

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
