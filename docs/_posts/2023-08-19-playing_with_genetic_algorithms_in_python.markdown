---
layout: post
title: "Playing with genetic algorithms in python"
date: 2023-08-19 00:00:00 -0000
categories: misc
---

A long long time ago I was young and going to school to attend a Computer Science program. It was so long ago that the data structures course was done in C++ and Java was shown to us as the new kid on the block that was going to revolutionize everything. Python was not mentioned, ever.

Something I learned in CS and that sticked in my head were Genetic Algorithms. I guess the reason was that GA were one of the first (and few) applied things I saw in CS and it seemed to me a simple, intuitive and brilliant idea. Today I was bored at home and I decided to play a little bit with it. 

[GAs](https://en.wikipedia.org/wiki/Genetic_algorithm) are a search technique that is inspired in biological genetic mutations and evolution which are used to purge certain parts of the search space. This is done encoding the search space into a genetic representation and a fitness function to evaluate each of the nodes.

I started implementing a useless but I think illustrative example of GAs which is generating a sequence of random bits and then search for it. I plot it as an $nxn$ matrix as it is easier for me to visualize and debug the process.

First of all I generate a random array of bits:

```python 
img = np.random.randint(2, size=(15,15))
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
```

![image](/assets/random_array.png)

This is a 15x15 array, so 225 bits and therefore a space of 2^225 possible combinations. Next I define the fitness function, which is how many bits of the image have the same value and in numpy it would be:

```python 
def score(matrix1, matrix2):
    return (matrix1 == matrix2).sum()
```

The transformations I use are:

* Crossover
* And mutation

And the entire code looks like below, where the parameters **population** is the initial population and **mutations** are the percentage of mutated bits.

```python 

def ga(img, population, mutations):

    def score(matrix1, matrix2):
        return (matrix1 == matrix2).sum()

    rows = img.shape[0]
    columns = img.shape[1]
    mid = rows//2

    mem = np.random.randint(2, size=(2 * population, rows, columns))
    scores = np.zeros((2 * population))

    for i in range(100000000):

        for k, matrix in enumerate(mem):
            scores[k] = score(matrix, img)

        max_score = np.argmax(scores)

        if scores[max_score] == rows * columns:
            print(i)
            plt.imshow(mem[max_score], cmap=plt.cm.gray)
            plt.show()
            break

        top_n_scores = np.argpartition(scores, -population)
        top = top_n_scores[-population:]
        bottom = top_n_scores[:-population]

        for j in range(population):
            
            r = random.randrange(len(top))  
            idx = [r, (r+1)%len(top)]
            parents = [top[idx[0]],top[idx[1]]]
            
            mem[bottom[j]][0:mid] = mem[parents[0]][0:mid]
            mem[bottom[j]][-(mid+1):] = mem[parents[1]][-(mid+1):]

            idx = np.random.choice([0,1], p=[(1-mutations), mutations],size=(rows,columns))
            mem[bottom[j]] = abs(mem[bottom[j]] - idx)

```



I think it was during the second year that we had two courses called Artificial Intelligence I and Artificial Intelligence II. There we learned all kind of things, and as far as I can remember among them there were a couple of slides somewhere about something called Neurons and how they would potentially be the future of AI. Holly molly.

Something else that we learned in that course and that sticked in my head were Genetic Algorithms. I guess the reason was that GA were one of the first (and few) applied things I saw in CS and it seemed to me a simple, intuitive and brilliant idea. Today I was bored at home and I decided to play a little bit with it. 

[GAs](https://en.wikipedia.org/wiki/Genetic_algorithm) are a search technique that is inspired in biological genetic mutations and evolution which are used to purge certain parts of the search space. This is done encoding the search space into a genetic representation and a fitness function to evaluate each of the nodes. 

I started implementing a useless but I think illustrative example of GAs which is generating a sequence of random bits and then search for it. I plot it as an $nxn$ matrix as it is easier for me to  



When I started the program I had absolutely no idea about Computer Science and I had never written a single line of code, the first year was mostly math and theory: Calculus I and II, Algebra I and II, Electronics (Physics) I and II... and some things like the C++ data structures class. 

, which meant everything was quite new for me, the first year was mostly maths and theory: Calculus I and II, Algebra I and II, Electronics (Physics) I and II... and some things like the C++ data structures. 

When I started the program I had absolutely no idea about Computer Science and I had never written a single line of code and I remember that one of the things that impacted were the Genetic Algorithms from one of the AI courses. I guess the reason was that GA were one of the first (and few) applied things I saw in CS and also was pretty intuitive. It seemed to me like a simple and brilliant idea.

In AI I and II we learned all kind of things and as far as I can remember, among them I remember a couple of slides about something called Neural Networks and how they would potentially be the future of AI. Holly molly.

