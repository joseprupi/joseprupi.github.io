---
layout: post
title: "Playing with genetic algorithms in python"
date: 2023-08-19 00:00:00 -0000
categories: misc
---

A long long time ago I was young and going to school to attend a Computer Science program. It was so long ago that the data structures course was done in C++ and Java was shown to us as the new kid on the block.

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

The genetic algorithm implementation per se (the fitness function, crossover and mutation) is implemented below, where the parameters **population** is the initial population and **mutations** are the percentage of mutated bits.

```python 

def ga(array, population, mutations):

    def score(matrix1, matrix2):
        return (matrix1 == matrix2).sum()

    rows = array.shape[0]
    columns = array.shape[1]
    mid = rows//2

    mem = np.random.randint(2, size=(2 * population, rows, columns))
    scores = np.zeros((2 * population))
    bottom = list(range(len(mem)))

    for i in range(1000000):
        
        # Bottom will contain all the random individuals generated when starting the execution
        # and the new individuals after the first iteration. Bottom means the bottom of the list
        # sorted by score
        for k in bottom:
            scores[k] = score(mem[k], array)

        # Check if the solution has been found
        max_score = np.argmax(scores)
        if scores[max_score] == rows * columns:
            print(i)
            plt.imshow(mem[max_score], cmap=plt.cm.gray)  # use appropriate colormap here
            plt.show()
            break

        # Select the population of individuals according to the score function
        top_n_scores = np.argpartition(scores, population)
        top = top_n_scores[population:]
        bottom = top_n_scores[:population]

        # Create #population new elements from the crossover and mutation
        for j in range(population):
            
            # Crossover -> Select parents from the top individuals
            #
            # I tried this with random choice and just picking a random position
            # from the top and the next one and the result is the same but way faster
            # It might be because of either the randomization of the initial population or maybe
            # the implementation of argpartition? or both?
            r = random.randrange(len(top))  
            idx = [r, (r+1)%len(top)]
            parents = [top[idx[0]],top[idx[1]]]
            
            mem[bottom[j]][0:mid] = mem[parents[0]][0:mid]
            mem[bottom[j]][-(mid+1):] = mem[parents[1]][-(mid+1):]

            # Mutation -> Mutate the bits
            #
            # The random choice of the bits to mutate is the most costly of the implementation
            # It seems there has to be some way to speed up this 
            idx = np.random.choice([0,1], p=[(1-mutations), mutations],size=(rows,columns))
            mem[bottom[j]] = abs(mem[bottom[j]] - idx)

```
With some trial error I ended up with an initial population of 500 and a mutation of 0.5% as the fastest way to find the array of bits, which finishes in approximately 160 iterations and 3 seconds in my computer. Even the experiment not making sense seems pretty good to me taking into account that the space is 2^225 and it can be implemented with few lines of code.

The entire implementation can be found [here](https://github.com/joseprupi/ga/blob/master/test.ipynb)

Next I wanted to try something more useful, solving mastermind with 6 colors and 4 selections, and in addition the original game had to be solved wth 12 choices.

