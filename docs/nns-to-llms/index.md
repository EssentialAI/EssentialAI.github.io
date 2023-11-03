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

    <div style="margin-top: -15px; text-align: center;">
        <a href="http://neuralnetworksanddeeplearning.com/chap1.html" target="_blank">Source: Neural Networks and Deep Learning</a>
    </div>

    <div style="margin-top: -12px; text-align: center;">[Edit Image](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1#R7ZlRb9owEMc%2FDY%2BdiJ0E8lig7V42Taqmtk%2BTlZjEmokzxylhn34XbEMc0pYiKJXYC%2Fj%2Bds7mfmdzhgGeLuo7SYrsm0goH6BhUg%2FwbIBQNBrBayOstBBEgRZSyRIteVvhnv2lRhwatWIJLZ2BSgiuWOGKschzGitHI1KKpTtsLrg7a0FSuiPcx4Tvqg8sUZlWx8Fwq3%2BlLM3szN7Q9CyIHWxclBlJxFJL6zH4ZoCnUgilW4t6SnkTOxsX7ej2hd7NwiTN1T4PfA%2FDQGTzx%2BL2QY1%2Fxk9z%2BsivjJdnwivzgc1i1cpGALxAsMGYLDOm6H1B4qZnCbhBy9SCg%2BVBk5SFJjBnNYVJJ7srtNNRqWjdksyK76hYUCVXMMT04tCE06QPioy9bMEwUtbiYDVi8Kcbz9sIQcME6R0BQ3sELE%2Bum8wDK%2BakLFnsxkmKKk%2BaAM2GYEFs5OqxbTy1jVntWCtjvRhaRWRK1dvEaeLk%2FS6AVoCDngBbTVJOFHt2d0tf1M0MPwSDFW%2F4bkBZvkEHXCkqGVPzVDu7O46w13EUdhzpwOw4WifB5mMfnhf%2BJ8%2BLz8J75GLC%2BFDenYMBo%2BhL8KHEg2MT%2F%2BzsfP9I7AL8hqMTkwv%2F79VDzmbfQwfu1RFyeX%2Fw2Tzu4R1yCNZkLtbr3IIP%2F1TCdlyV62r0GgZ446JeE7P90Er1Oxz8fv3L0w3rF5apXdtRnfSC6ke5CVUqKX7TqeBCgpKLvCm45ozzjkQ4S%2FMmKyFpKOiTppZiUK9em44FSxL%2BUrW254Gzf32GsO9%2B7aLd%2Bix8JUWPXp9Fp2eNLpU1CtxtPETnZW1LvlPCxpcKu7Oxcc%2FF62Nh911Vjwp7ebGnOEady5N%2Fbth91%2Bwjw77YYzzqfGWffWfj08O%2B1GO8u7NxGJ4Zdt%2B1%2BZ2w%2FVdhD4LpGt9oIipVVNCYXSj87oUd2d%2FeW%2FCjHvj%2B%2B%2BGDuf0lW9%2Fbtn8H4Jt%2F){:target="_blank"}</div>

    </div>

The output from the perceptron

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









