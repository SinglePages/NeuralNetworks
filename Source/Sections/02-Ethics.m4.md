# Ethics {#sec:ethics}

> A bit of the maker goes into that which they make.
>
> -- Unknown


How is ethics important to AI? It can help us answer questions such as:

- What should we build?
- What should we **not** build?
- How should we build something?

m4question([[Who is in charge of enforcing ethics in AI?]], [[Everyone and no one. We do not have a special ethics force to guide us. The problem is clearly that if everyone is responsible, nobody will think they need to act.]])

m4question([[When should you start to consider ethical implications?]], [[From the very beginning. This will make it easier to:

- avoid pitfalls,
- analyze results from an ethical lens,
- avoid wasting time, and
- ensure the system **is** ethical.
]])

Ethics should be easy, but it is hard because we all come to the table with our own value systems, opinions, motivations, and power. What would you do if you were directed to build something you knew to be unethical? How does your answer change if your choices are to build or to quit?

## Key Topics

This is not going to be an exhaustive discussion on ethics in AI and NNs. Instead, I'll point you to the resources in the [Additional Material]($additional-material-1) section. The topics below are taken from [Ethics — fastbook](https://github.com/fastai/fastbook/blob/master/03_ethics.ipynb "fastbook/03_ethics.ipynb at master · fastai/fastbook").

Topics to consider:

1. **Recourse and accountability**: who is responsible (and liable) for the developed system? The user, developer, manager, owner, company, other?
2. **Feedback loops**: does the system control creation of the next round of input data (such as a video recommendation system)?
3. **Bias**: all systems have bias; what bias is in your system? Is the source of bias historical, from measurement, from aggregation, from the representation, other?
4. **Disinformation**: can your system be used for nefarious goals?

## Strategies

Here are some questions you can ask to prevent running into trouble (from [Practical Data Ethics](https://ethics.fast.ai)):

- Should we even be doing this?
- What bias is in the data? (All data contains bias.)
- Can the code and data be audited?
- What are errors rates for different sub-groups?
- What is the accuracy of a simple rule-based alternative?
- What processes are in place to handle appeals/mistakes?
- How diverse is the team?

When should you ask these questions? The Markkula Center for Applied Ethics recommends scheduling regular meeting in which you perform ethical risk sweeping. See their [Ethical Toolkit](https://www.scu.edu/ethics-in-technology-practice/ethical-toolkit/ "Ethical Toolkit - Markkula Center for Applied Ethics") for more information.

## Additional Material

- [Practical Data Ethics](https://ethics.fast.ai)
- [Fair ML Book](https://fairmlbook.org/)
- [Machine Ethics Podcast](https://www.machine-ethics.net/podcast/)
- Codes of Ethics from the [ACM](https://www.acm.org/code-of-ethics), [IEEE](https://www.ieee.org/about/corporate/governance/p7-8.html), and the [Data Science Association](https://www.datascienceassn.org/code-of-conduct.html)
