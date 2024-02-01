# LEARNING to make decisions
In the previous section we look at the basic perceptron model and a simplified decision making framework.

However, we could not move on to making decisions (for unseen scenarios) as we still have to find the optimal values for weights $w_i$ and bias $b$. In this chapter, we shall look exactly how these perceptron/neural network models learn.

# Making Perceptron Learn

We have, with us a perceptron model, that makes binary decisions as follows:

Using `0` for YES and `1` for NO.

$$
\text{output } = \left\{ \begin{matrix}
\text{0 if } \sum_j x_j w_j - b \geq 0\\ 
\text{1 if } \sum_j x_j w_j - b < 0\\
\end{matrix}\right.
$$

The two parameters, above that control the output of the model are `weights` and `bias`. **Changing values of weights and biases changes the output of the model (perceptron).** If we would like to make an optimized models, we have to learn the optimal values of `weights` and `biases` for a given problem at hand for a given dataset.

In the above setting of perceptron we see that changing values of weights and bias can lead to any values on the real line output and with many features, slight changes in weights can lead to vast changes in the output, lets fix that using some function (called activation function) on top of the weighted output.

## Sigmoid Activation

Sigmoid (also called Logistic) function takes any input on the real line and returns value between 0 to 1. This works really well for our model as we can be sure that, no matter what the weights are, we know that the output of our model will be between 0 to 1. **Importantly, this helps us to set a threshold of `0.5` to make decision.** This is a very commonly accepted threshold in the field of Deep Learning.


$$
\text{sigmoid(z)} = \sigma(z) = \frac{1}{1+e^{-z}}
$$

In our case, the perceptron output becomes

$$
\text{output} = \text{sigmoid} ( \sum_j x_jw_j )
$$

Adding our threshold of `0.5` into the equation

$$
\text{output } = \left\{ \begin{matrix}
\text{1 if } \sigma(\sum_j x_j w_j - b) \geq 0\\ 
\text{0 if } \sigma(\sum_j x_j w_j - b) < 0\\
\end{matrix}\right.
$$

`b is replaced with -b in most books/reference material for simplicity as b is learnt by the model itself.`

!!! info "Why Sigmoid?"
    Keep in mind that sigmoid activation function is not a core requirement to make the perceptron model LEARN. Sigmoid is an activation function that is added to the output of each perceptron (node) to make sure the output of the perceptron falls between 0 and 1, allowing us to set a threshold of 0.5 to make decisions.

    **Our perceptron model has not started learning yet.**

!!! danger "But does using sigmoid function change the way perceptron is modeled?"
    Let's take a look at the Sigmoid curve. To make sure that an inequality is maintained before and after a function is applied, there are some checks required to be passed.

    1. The function must be monotonically increasing or decreasing.
    2. The function must be defined everywhere on the real number line.

    <figure markdown>
        ![](../../assets/images/from-nns-to-llms/sigmoid.png)
    </figure>

    From the above figure we see that sigmoid function satisfies these conditions, which means adding sigmoid function does not effect the perceptron decision making inequality.

    In addition to this:
    
    1. 
