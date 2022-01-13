# NeuralNetworks

- encourage implementing alongside
- show how to do so with visual studio code

Resources
- https://pythonalgos.com/2021/12/06/create-a-neural-network-from-scratch-in-python-3/

Ideas

- add "additional resources" to each section
- add pointer to BCE and why it is better than MSE
- multiple programming languages (tabs?)
- single page and multi-page options
- prerender math during build
- add comments column for course?
- list of tags per section
- load other content on click in iframes
	+ set source to "" then change on click
- add index with definitions
- connect variables and diagrams so that when you hover it will show in multiple places
- use roughjs?
- svg line label background...
- use image symlinks?

https://thoughtspile.github.io/grafar/#/

https://setosa.io/#/
interactive = intuitive
substance > information

Theme



⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨


https://github.com/observablehq/plot

Tools

- [Convert LaTeX expressions to Unicode](https://www.unicodeit.net/)
- create interactive demos with
	+ https://github.com/karpathy/recurrentjs
	+ https://github.com/karpathy/convnetjs
	+ https://github.com/karpathy/reinforcejs
	+ https://cs.stanford.edu/people/karpathy/gan/


https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network

Animation:

```python
with open('animation.html', 'w') as html_file:
    html_file.write(animation.to_jshtml())
```


Docs:

- https://lierdakil.github.io/pandoc-crossref/#customization
- https://docs.mathjax.org/en/latest/web/typeset.html

Resources:

- [colah's blog](https://colah.github.io/)
	+ many articles on NN
- [Interactive neural network training](https://playground.tensorflow.org/)
- [Distill](https://distill.pub/)
	+
- [Explorable Explanations](http://worrydream.com/ExplorableExplanations/)
	+ create reactive examples
- [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [Model Tuning and the Bias-Variance Tradeoff](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
- [Lil'Log](https://lilianweng.github.io/lil-log/)
	+ Assortment of articles about NNs
- [off the convex path](https://www.offconvex.org/)
- [Setosa]https://setosa.io/#/
	+ Visual explanations

https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
https://distill.pub/2017/feature-visualization/
https://distill.pub/2017/ctc/ (sequence modeling with CTC)
https://distill.pub/2016/augmented-rnns/

https://cs231n.github.io/


[DIFFERENTIAL EQUATIONS EXPLAINED](https://lewis500.github.io/diffeq/)
[Ordinary Least Squares Regression](https://setosa.io/ev/ordinary-least-squares-regression/)



- fork or alter output of ansifilter
- border instead of pipe
- use bat instead of pandoc highlighter?
- ansifilter -> table-cell last one width 100%


individual pages
- fix relative links
- add TOC


old svg method:
esyscmd(dot -Tsvg Diagrams/SingleNeuron.dot | tail -n +4)


https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/


Computing memory usage
- optimization issues (use nvidia-smi)
- parameters (sometimes duplicated for certain computations)
- activations and inputs
- batches
https://cs231n.github.io/convolutional-networks/#computational-considerations
