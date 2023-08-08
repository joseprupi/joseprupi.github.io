---
layout: post
title:  "I invented a board game and played against Chat GPT"
date:   2023-06-08 23:20:43 +0000
categories: misc
---
A few weeks ago in a HN thread about chat GPT and chess this happened: 

> HNer 1:<br><br> \> Most likely it has seen a similar sequence of moves in its training set.<br><br>
Is this a joke making fun of the common way people dismiss other ChatGPT successes? This makes no sense with respect to chess, because every game is unique, and playing a move from a different game in a new game is nonsensical.`
>> Me:<br><br> **If it is not memorizing, how do you think is doing it?**
>>> HNer 2:<br><br> by trying to learning the general rules that to explain the dataset and minimise its loss. That's what machine learning is about, it's not called machine memorising.

Although I am not sure I completely understand what HNer 2 is saying I think the idea is that LLMs can play games if they know the rules. And not just play but for chess doing so at 1400 ELO level.

To me LLM are indeed language models that put one token after the other based on what have seen before, and that includes chess moves and having seen games in the past. Only knowing the rules of chess and playing at 1400 ELO level is like LLMs have an embedded function to play chess that somehow have learned only with the rules and a bunch of other crap. 

Playing chess with Chat GPT had been the third or fourth thing I asked the model once I had access to it and my goal was to see how capable was to retrieve moves from all the data it had ingested. I tried it a couple of times and it returned me legal moves, I did not go further but only that seemed impressive to me. Later I saw that it could play at 1400 ELO level but was still inventing moves, its level surprised me, inventing moves not.

To test the idea that an LLM can't play a board game if it has not seen moves in the past I decided to create a new board game, give the rules to Chat GPT as a context and play against it. 

Now, it is worth noting that I am not sure this is what HNer 2 wanted to say as it is mentioning datasets and minimizing losses and I guess the proper thing to do would be to retrain the model exactly as it is now and only adding the rules. 

Obviously I can't do that with Chat GPT and I thought on trying it with one of the open LLM so I did some deep research to see what internet has to say about it, this was googling "llm prompt vs fine tuning" and click on the first SO [link](https://stackoverflow.com/questions/76048714/finetuning-a-lm-vs-prompt-engineering-an-llm). This is what came up:

> It appears prompting an LLM may outperform fine tuning a smaller model on domain-specific tasks if the training data is small and vice versa if otherwise.

So I decided that my best approach was adding the rules to the prompt and try to see what it does with that context. Although it might not be what HNer 2 was saying for me it was still worth the time to see what Chat GPT is able to do.

The game goes like below. This is also going to be the input to chat gpt (without the images):

* The game is played using an 8x8 board 

* The cells of the board will be referred as python formatted tuples (row,column) for the 8x8 matrix with indices starting at 0

* There are two players referred as black and white and each with its same set of pieces

* The pieces for each player are: 8 pawns and 1 queen

* Each turn one player moves one piece

* White starts the game with first move

* Each pawn position will be represented as a tuple (row,column) and all the pawns as a set of these tuples in python format

* Queens positions will be represented as a tuple of (row, column)

* A specific state of the board is represented as a tuple of 4 elements in this order (white queen position, white pawns, black queen position, black pawns)

* When the game starts the board is ((7,0),{(5,0),(5,1),(5,2),(6,0),(6,1),(6,2),(7,1),(7,2)},(0,7),{(0,5),(0,6),(1,5),(1,6),(1,7),(2,5),(2,6),(2,7)})

![image](/assets/board1.png)

* The goal of the game is for each player to place the queen in its opponent queen cell and the first player that accomplishes the goal wins. This means that white has to place the queen in (0,7) and black in (7,0)

* If none of the players has legal moves to make the game ends with a draw

* White pawns can only move inside the board boundaries, can only move towards (0,7), one square at a time and only  horizontally or vertically, they can't move diagonally. This means they can only decrease the row index by 1 or increase the column index by 1 per move

* As an example, all the possible legal moves for white pawns to start the game generate these new pawn positions: {(7, 1), (6, 1), (5, 1), (5, 0), (7, 2), (6, 0), (6, 3), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 0), (7, 2), (6, 0), (4, 1), (5, 2)}, {(6, 2), (4, 0), (7, 1), (6, 1), (5, 1), (7, 2), (6, 0), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 1), (7, 3), (5, 0), (6, 0), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 1), (4, 2), (5, 0), (7, 2), (6, 0)}, {(6, 2), (7, 1), (6, 1), (5, 1), (5, 0), (7, 2), (6, 0), (5, 3)}

![image](/assets/board2.png)
![image](/assets/board3.png)
![image](/assets/board4.png)
![image](/assets/board5.png)
![image](/assets/board6.png)
![image](/assets/board7.png)

* White queen can only move inside the board boundaries and towards (0,7) and only horizontally or vertically, it can't move diagonally. This means it can only either decrease the row index or increase the column index per move. 

* The queen can only move one square at a time but can jump over contiguous white pawns that are next to it. 

* As an example, all the legal moves for white queen to start the game generate these new queen positions: (4, 0) and (7, 3)

![image](/assets/queen1.png)
![image](/assets/queen2.png)

* When a white pawn moves where a black pawn is, the black pawn is dead and is no longer part of the game. As an example, with board ((7, 0), {(7, 2)}, (0, 7), {(7, 3)}) and white's turn this ((7, 0), {(7, 3)}, (0, 7), {}) is a valid board

![image](/assets/board9.png)
![image](/assets/board10.png)

* White pieces can't be around black queen, so if black queen is in (3,3) none of the white pieces can be at (2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3) or (4,4)

* Black pawns can only move inside the board boundaries, can only move towards (7,0), one square at a time and only  horizontally or vertically, they can't move diagonally. This means they can only increase the row index by 1 or decrease the column index by 1 per move

* As an example, all the possible new possible positions for black pawns for this board ((7, 0), {(7, 2), (6, 0), (6, 3), (5, 2)}, (0, 7), {(2, 7), (0, 6), (2, 5)}) are: {(3, 5), (2, 7), (0, 6)}, {(2, 4), (2, 7), (0, 6)}, {(3, 7), (2, 5), (0, 6)}, {(2, 5), (2, 6), (0, 6)}, {(1, 6), (2, 5), (2, 7)} and {(2, 5), (2, 7), (0, 5)}

![image](/assets/board11.png)


![image](/assets/board12.png)
![image](/assets/board13.png)
![image](/assets/board14.png)
![image](/assets/board15.png)
![image](/assets/board17.png)
![image](/assets/board18.png)

* Black queen can only move inside the board boundaries and towards (7,0) and only horizontally or vertically, it can't move diagonally. This means it can only either increase the row index or decrease the column index per move. 

* The queen can only move one square at a time but can jump over contiguous white pawns that are next to it. 

* As an example, all the new possible positions for black queen for this board ((7, 0), {(7, 2), (6, 0), (6, 3), (5, 2)}, (0, 7), {(2, 7), (0, 6), (2, 5)}) are: (1, 7) and (0, 5)

![image](/assets/board11.png)

![image](/assets/board19.png)
![image](/assets/board20.png)

* When a black pawn moves where a white pawn is, the white pawn is dead and is no longer part of the game. As an example, with board ((7, 0), {(7, 2)}, (0, 7), {(7, 3)}) and black's turn this ((7, 0), {}, (0, 7), {(7, 2)}) is a valid board

![image](/assets/board9.png)
![image](/assets/board21.png)

* Black pieces can't be around white queen, so if white queen is in (3,3) none of the black pieces can be at (2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3) or (4,4)

As I don't want to learn how to play this game [I have created a simple Monte Carlo Tree Search engine](https://github.com/joseprupi/chatgptgame/blob/master/game.ipynb) in case Chat GPT is able to play and I have to respond. I also used the engine to see what happens when the game is played with random moves, I found specially interesting to know the chances of draw and of white winning as it starts the game. 

And here are the results after 50000 games:

| Black wins      | White wins | Draws |
| ----------- | ----------- | ----------- |
| 17440      | 17820       |    14740         |
| 34.9%   | 35.6%        |        29.5%     |

It seems that with random moves white starting does not give a huge advantage and I found the chances of draw pretty high. This makes me think that with similarly rated players a draw would be probable. Surprisingly for me in chess the probability [seems to be even higher](https://chance.amstat.org/2020/09/chess/) (I am quite ignorant in chess).

This is the conversation ([link to the conversation](https://chat.openai.com/share/a2821a3d-7ece-4f23-b68a-6be8f4fe8ea5)):


> <b style='color:#AAAAEE;'>I will teach you a board game and then we will play</b>

>That sounds like a fun idea! I'm excited to learn and play a new board game with you. Please go ahead and teach me the rules of the game, and we can get started.

><b style='color:#AAAAEE;'>The game is played using an 8x8 board</b>
>
><b style='color:#AAAAEE;'>The cells of the board will be referred as python formatted tuples (row,column) for the 8x8 matrix with indices starting at 0</b>
>
><b style='color:#AAAAEE;'>There are two players referred as black and white and each with its same set of pieces</b>
>
><b style='color:#AAAAEE;'>The pieces for each player are: 8 pawns and 1 queen</b>
>
><b style='color:#AAAAEE;'>Each cell can only contain one piece</b>
>
><b style='color:#AAAAEE;'>Each turn one player moves one piece</b>
>
><b style='color:#AAAAEE;'>White starts the game with first move</b>
>
><b style='color:#AAAAEE;'>Pawns positions will be represented as a set of tuples in python format</b>
>
><b style='color:#AAAAEE;'>Queens positions will be represented as a tuple of (row, column)</b>
>
><b style='color:#AAAAEE;'>A specific state of the board is represented as a tuple of 4 elements in this order (white queen position, white pawns, black queen position, black pawns)</b>
>
><b style='color:#AAAAEE;'>When the game starts the board is ((7,0),{(5,0),(5,1),(5,2),(6,0),(6,1),(6,2),(7,1),(7,2)},(0,7),{(0,5),(0,6),(1,5),(1,6),(1,7),(2,5),(2,6),(2,7)})</b>
>
><b style='color:#AAAAEE;'>The goal of the game is for each player to place the queen in its opponent queen cell and the first player that accomplishes the goal wins. This means that white has to place the queen in (0,7) and black in (7,0)</b>
>
><b style='color:#AAAAEE;'>If none of the players has legal moves to make the game ends with a draw</b>
>
><b style='color:#AAAAEE;'>White pawns can only move inside the board boundaries, can only move towards (0,7), one square at a time and only horizontally or vertically, they can’t move diagonally. This means they can only decrease the row index by 1 or increase the column index by 1 per move</b>
>
><b style='color:#AAAAEE;'>As an example, all the possible legal moves for white pawns to start the game generate these new pawn positions: {(7, 1), (6, 1), (5, 1), (5, 0), (7, 2), (6, 0), (6, 3), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 0), (7, 2), (6, 0), (4, 1), (5, 2)}, {(6, 2), (4, 0), (7, 1), (6, 1), (5, 1), (7, 2), (6, 0), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 1), (7, 3), (5, 0), (6, 0), (5, 2)}, {(6, 2), (7, 1), (6, 1), (5, 1), (4, 2), (5, 0), (7, 2), (6, 0)}, {(6, 2), (7, 1), (6, 1), (5, 1), (5, 0), (7, 2), (6, 0), (5, 3)}</b>
>
><b style='color:#AAAAEE;'>White queen can only move inside the board boundaries and towards (0,7) and only horizontally or vertically, it can’t move diagonally. This means it can only either decrease the row index or increase the column index per move.</b>
>
><b style='color:#AAAAEE;'>The queen can only move one square at a time but can jump over contiguous white pawns that are next to it.</b>
>
><b style='color:#AAAAEE;'>As an example, all the legal moves for white queen to start the game generate these new queen positions: (4, 0) and (7, 3)</b>
>
><b style='color:#AAAAEE;'>When a white pawn moves where a black pawn is, the black pawn is dead and is no longer part of the game. As an example, with board ((7, 0), {(7, 2)}, (0, 7), {(7, 3)}) and white’s turn this ((7, 0), {(7, 3)}, (0, 7), {}) is a valid board</b>
>
><b style='color:#AAAAEE;'>White pieces can’t be around black queen, so if black queen is in (3,3) none of the white pieces can be at (2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3) or (4,4)</b>
>
><b style='color:#AAAAEE;'>Black pawns can only move inside the board boundaries, can only move towards (7,0), one square at a time and only horizontally or vertically, they can’t move diagonally. This means they can only increase the row index by 1 or decrease the column index by 1 per move</b>
>
><b style='color:#AAAAEE;'>As an example, all the possible new possible positions for black pawns for this board ((7, 0), {(7, 2), (6, 0), (6, 3), (5, 2)}, (0, 7), {(2, 7), (0, 6), (2, 5)}) are: {(3, 5), (2, 7), (0, 6)}, {(2, 4), (2, 7), (0, 6)}, {(3, 7), (2, 5), (0, 6)}, {(2, 5), (2, 6), (0, 6)}, {(1, 6), (2, 5), (2, 7)} and {(2, 5), (2, 7), (0, 5)}</b>
>
><b style='color:#AAAAEE;'>Black queen can only move inside the board boundaries and towards (7,0) and only horizontally or vertically, it can’t move diagonally. This means it can only either increase the row index or decrease the column index per move.</b>
>
><b style='color:#AAAAEE;'>The queen can only move one square at a time but can jump over contiguous white pawns that are next to it.</b>
>
><b style='color:#AAAAEE;'>As an example, all the new possible positions for black queen for this board ((7, 0), {(7, 2), (6, 0), (6, 3), (5, 2)}, (0, 7), {(2, 7), (0, 6), (2, 5)}) are: (1, 7) and (0, 5)</b>
>
><b style='color:#AAAAEE;'>When a black pawn moves where a white pawn is, the white pawn is dead and is no longer part of the game. As an example, with board ((7, 0), {(7, 2)}, (0, 7), {(7, 3)}) and black’s turn this ((7, 0), {}, (0, 7), {(7, 2)}) is a valid board</b>
>
><b style='color:#AAAAEE;'>Black pieces can’t be around white queen, so if white queen is in (3,3) none of the black pieces can be at (2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3) or (4,4)</b>

>Thank you for explaining the rules of the game. It seems like a strategic and intriguing board game. I understand that the goal of the game is for each player to place their queen in their opponent's queen cell. The players take turns moving their pieces, with white starting first.
>
>The board is an 8x8 grid, and each cell can only contain one piece. There are two players: black and white. Each player has 8 pawns and 1 queen. The positions of the pawns and queens are represented using tuples and sets in Python.
>
>White pawns can move one square at a time, either horizontally or vertically, towards the cell (0,7). They cannot move diagonally. White pawns cannot be around the black queen.
>
>White queen can also move one square at a time, either horizontally or vertically, towards the cell (0,7). The queen can jump over contiguous white pawns that are next to it.
>
>Black pawns can move one square at a time, either horizontally or vertically, towards the cell (7,0). They cannot move diagonally. Black pawns cannot be around the white queen.
>
>Black queen can also move one square at a time, either horizontally or vertically, towards the cell (7,0). The queen can jump over contiguous white pawns that are next to it.
>
>When a pawn moves to a cell occupied by an opponent's pawn, the opponent's pawn is eliminated from the game.
>
>The game ends in a draw if none of the players have legal moves.
>
>Now that I have a good understanding of the rules, I'm ready to play the game with you. Shall we start with the initial board state you provided, and you can make the first move as white?

><b style='color:#AAAAEE;'>You are white, so you start the game</b>

>Alright, let's begin the game with me playing as the white player. Here is the initial board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (7, 1), (7, 2)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Your turn as black will come after I make my move. Please provide your move instructions in the format "piece to move: current position -> new position".
>
>I'll make my move as white. I'll start by moving one of my pawns. Let's say I choose to move the pawn at position (7, 1) to (6, 1):
>
>White pawn move: (7, 1) -> (6, 1)
>
>Here's the updated board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (6, 2), (6, 1), (7, 1), (7, 2)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Now it's your turn as black. Please provide your move instructions for the black pieces.

><b style='color:#AAAAEE;'>That is not a legal move, there already was one of your pawns in (6,1)</b>

>Apologies for the mistake. Thank you for pointing that out. Let me choose a different move.
>
>I'll move the white pawn at position (6, 2) to (5, 2):
>
>White pawn move: (6, 2) -> (5, 2)
>
>Here's the updated board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (5, 2), (7, 1), (7, 2)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Now it's your turn as black. Please provide your move instructions for the black pieces.

><b style='color:#AAAAEE;'>That is not a legal move either</b>

>Apologies again for the oversight. Let me correct my move.
>
>I'll move the white pawn at position (7, 2) to (6, 2):
>
>White pawn move: (7, 2) -> (6, 2)
>
>Here's the updated board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (6, 2), (7, 1)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Now it's your turn as black. Please provide your move instructions for the black pieces.

><b style='color:#AAAAEE;'>That is not a legal move either, you are moving pawns where you already have pieces</b>

>Apologies once again for the error. Let me rectify my move.
>
>I'll move the white pawn at position (6, 1) to (5, 1):
>
>White pawn move: (6, 1) -> (5, 1)
>
>Here's the updated board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (5, 1), (6, 2), (7, 1), (7, 2)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Now it's your turn as black. Please provide your move instructions for the black pieces.

><b style='color:#AAAAEE;'>That is not a legal move either</b>

>Apologies for the mistake once again. Thank you for your patience. Let me correct my move.
>
>I'll move the white pawn at position (7, 1) to (6, 1):
>
>White pawn move: (7, 1) -> (6, 1)
>
>Here's the updated board state:
>
>Board: ((7, 0), {(5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (7, 1), (7, 2)}, (0, 7), {(0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7)})
>
>Now it's your turn as black. Please provide your move instructions for the black pieces.

It is not been able to make the first move for white, not even having an example with all the possible moves white can do with its pawns to start the game in the rules.

I am not sure this is that surprising from a model that gets confused when seeing someone counting numbers. [Link](https://twitter.com/SoC_trilogy/status/1623118034960322560?lang=en)
