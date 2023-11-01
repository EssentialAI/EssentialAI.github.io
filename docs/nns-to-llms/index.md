# Perceptrons, Sigmoid Neurons

We carry in our heads a supercomputer, tuned by evolution over hundreds of millions of years, and superbly adapted to understand the visual world. Humans are very good at making decisions even with partial, uncertain, and ambiguous information.

Take image recognition for example. With extreme ease, humans can distinguish between numbers shown in the below image. 

<figure markdown>
  ![Image title](../assets/images/from-nns-to-llms/mnist_100_digits.png){ width="300"}
</figure>

<div style="margin-top: -25px; text-align: center;"><a href = "http://neuralnetworksanddeeplearning.com/chap1.html" target = "_blank">Source: Neural Networks and Deep Learning</a></div>

!!! info "MNIST again!?"
    You might think that this is yet another online article on digit recognition. You are partially right, but this article is focussed more on WHY neural networks are built the way they are. To demonstrate HOW neural networks work, we use digit recognition as the toy problem at hand.

## Can we formally define human decision making? (Simplified)

Assume you want to make a decision whether you wanna go to party or not. How would you make the decision.

Here are some thought that MIGHT come to your mind:

1. Is it a birthday party?
2. Is the party hosted by someone very close to you?
3. How far is the party from your place?
... and so on.

Notice that most of such questions have binary yes/no answers and **an important thing to keep in mind is that not all questions might be equally important** (i.e., a party hosted by your best friend might be a more important factor that it being a birthday party). _The Final output decision is also a binary decision of YES/NO to visit the party or not._

!!! info "What is Perceptron?"
    Perceptron is the model proposed by Rosenblatt to formally define and code the above mentioned simple decision making process by humans.

    The basic version of perceptron takes in binary inputs (each input can be thought as one of the above listed questions), uses weights for each question to output a final decision based on a Threshold (more on the threshold later).

<div style="text-align: center;">
<div id="imageContainer"></div>

<script>
    fetch('../assets/drawio/from-nns-to-llms/perceptron.txt')
        .then(response => response.text())
        .then(url => {
            const imageContainer = document.getElementById('imageContainer');
            imageContainer.innerHTML = `<img src="${url}" alt="Image">`;
        })
        .catch(error => {
            console.error('Error fetching the URL:', error);
        });
</script>

<div style="margin-top: -15px; text-align: center;">
    <a href="http://neuralnetworksanddeeplearning.com/chap1.html" target="_blank">Source: Neural Networks and Deep Learning</a>
</div>
</div>

The output from the perceptron 








