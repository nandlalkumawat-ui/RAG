# Evaluation results

Average retrieval precision@3: **1.00**

## What is the difference between breadth-first search and depth-first search?

Expected pages 8-11 | retrieval precision: 3/3 = 1.00

**Answer** (72 s):

According to [1] and [2], the main difference between Breadth-First Search (BFS) and Depth-First Search (DFS) is:

* BFS always selects the shallowest frontier node from the start node for expansion, using a first-in, first-out (FIFO) queue to represent the frontier. This ensures that nodes are visited in order of their insertion.
* DFS does not guarantee this ordering, instead selecting the "leftmost" solution in the search tree without regard for path costs.

In other words, BFS explores all nodes at a given depth level before moving on to the next level, while DFS explores as far as possible along each branch before backtracking.

**Sources:**

- [1] book_1.pdf, page 9, chunk 2 (similarity 0.6582): • Description - Breadth-first search is a strategy for exploration that always selects theshallowest frontier node from the start node for expansion. • Frontier Representation - If we want to visit sh...
- [2] book_1.pdf, page 9, chunk 1 (similarity 0.6374): faithfully yet tragically get “stuck” searching for the deepest node in an infinite-sized search tree, doomed to never find a solution. • Optimality - Depth-first search simply finds the “leftmost” so...
- [3] book_1.pdf, page 8, chunk 2 (similarity 0.5961): states in our search tree, we are forced to select our strategy for tree search from one of the techniques that falls under the umbrella ofuninformed search. We’ll now cover three such strategies in s...

## What makes a heuristic admissible?

Expected pages 12-16 | retrieval precision: 3/3 = 1.00

**Answer** (39 s):

According to [1], a heuristic is considered admissible if it satisfies the condition:

∀n, 0≤ h(n)≤ h∗(n)

In other words, the value of the heuristic for any given state must be less than or equal to the true optimal forward cost to reach that goal state from the node.

**Sources:**

- [1] book_1.pdf, page 15, chunk 2 (similarity 0.587): trivial heuristic. The trivial heuristic is often incorporated at the base of asemi-lattice for a search problem, a dominance hierarchy of which it is located at the bottom. Below is an example of a s...
- [2] book_1.pdf, page 16, chunk 0 (similarity 0.5501): 16 Figure 3: Semi-lattice example same range. It’s common practice to generate multiple admissible heuristics for any given search problem and compute the max over the values output by them to generat...
- [3] book_1.pdf, page 14, chunk 0 (similarity 0.5094): 14 Defining h∗(n) as the true optimal forward cost to reach a goal state from a given node n, we can formulate the admissibility constraint mathematically as follows: ∀n, 0≤ h(n)≤ h∗(n) Theorem. For a...

## How does alpha-beta pruning work?

Expected pages 47-53 | retrieval precision: 3/3 = 1.00

**Answer** (61 s):

Alpha-beta pruning works by stopping the search as soon as it is determined that the value of a node can at best equal the optimal value of its parent. In other words, if the minimum possible value of a node's successors is less than or equal to the maximum possible value of its parent's successors, then there is no need to explore further. This optimization reduces the number of nodes that need to be evaluated, resulting in a faster search process. [1]

**Sources:**

- [1] book_1.pdf, page 50, chunk 1 (similarity 0.6451): , this yields far too great a runtime for many games. For example, chess has a branching factorb≈ 35 and tree depth m≈ 100. To help mitigate this issue, minimax has an optimization -alpha-beta pruning...
- [2] book_1.pdf, page 52, chunk 0 (similarity 0.5721): 52 the player who thinks 2 moves ahead. This pruning is exactly what the minimax algorithm with alpha-beta pruning does, and is implemented as follows: Figure 9: Alpha-Beta Pseudocode Take some time t...
- [3] book_1.pdf, page 50, chunk 0 (similarity 0.5002): 50 In implementation, minimax behaves similarly to depth-first search, computing values of nodes in the same order as DFS would, starting with the leftmost terminal node and iteratively working its wa...

## What is the Bellman equation?

Expected pages 64-75 | retrieval precision: 3/3 = 1.00

**Answer** (55 s):

The Bellman equation for Markov decision processes is defined as:

V∗(s) = max a ∑ s′T (s,a,s′)[R(s,a,s′) +γV∗(s′)]

[1], [2]

**Sources:**

- [1] book_1.pdf, page 68, chunk 1 (similarity 0.5989): . Using these two new quantities and the other MDP quantities discussed earlier, the Bellman equation is defined as follows: V∗(s) = max a ∑ s′T (s,a,s′)[R(s,a,s′) +γV∗(s′)] Before we begin interpreti...
- [2] book_1.pdf, page 68, chunk 0 (similarity 0.5129): 68 Start State Reward c 0.1 d 0.1 e 1 We’ll now learn how to solve such MDPs (and much more complex ones!) algorithmically using theBellman equation for Markov decision processes. 4.2.1 The Bellman Eq...
- [3] book_1.pdf, page 68, chunk 3 (similarity 0.5108): take another step outwards and consider the full equation for Q-value. Knowing[R(s,a,s′) + γV∗(s′)] represents the utility attained by acting optimally after arriving in states′ from Q-state (s,a ), i...

## What is the trade-off between exploration and exploitation?

Expected pages 78-89 | retrieval precision: 3/3 = 1.00

**Answer** (50 s):

The trade-off between exploration and exploitation is that increasing exploration (e.g., using a higher value of ϵ) leads to more frequent but less informed actions, while decreasing exploration (e.g., using a lower value of ϵ) leads to fewer but more informed actions. [1]

**Sources:**

- [1] book_1.pdf, page 88, chunk 0 (similarity 0.5068): 88 5.4 Exploration and Exploitation 5.4 Exploration and Exploitation We’ve now covered several different methods for an agent to learn an optimal policy, and emphasized that “sufficient exploration” i...
- [2] book_1.pdf, page 88, chunk 2 (similarity 0.4912): been visited. Agents in a states always select the action that has the highestf (s,a ) from each state, and hence never have to make a probabilistic decision between exploration and exploitation. Inst...
- [3] book_1.pdf, page 88, chunk 1 (similarity 0.4532): algorithm) to learn the optimal policy very slowly. To get around this,ϵ must be manually tuned and lowered over time to see results. 5.4.2 Exploration Functions The issue of manually tuningϵ is avoid...

## How does particle filtering work?

Expected pages 140-142 | retrieval precision: 3/3 = 1.00

**Answer** (59 s):

Particle filtering is a simulation that begins with particle initialization, which can be done flexibly. The simulation takes on a similar form to the forward algorithm, with two main updates at each timestep:

1. Time Elapse Update: Updates the value of each particle according to the transition model, sampling from the probability distribution given by P (Ti+1| ti).
2. Observation Update: Weights each particle according to the probability dictated by the observed evidence and the particle's state, using the sensor model P (Fi| Ti).

The process involves calculating weights for all particles, normalizing the total weight over states, and resampling particles from this distribution if necessary.

[1], [2]

**Sources:**

- [1] book_1.pdf, page 140, chunk 3 (similarity 0.5296): that we’ve seen how to recover a belief distribution from a particle list, all that remains to be discussed is how to generate such a list for a timestep of our choosing. 8.4.1 Particle Filtering Simu...
- [2] book_1.pdf, page 140, chunk 1 (similarity 0.5015): as the Forward Algorithm: it gives us an approximation of P (XN| e1:N ). Instead of storing a full probability table mapping each state to its belief probability, we’ll instead store a list of n parti...
- [3] book_1.pdf, page 141, chunk 0 (similarity 0.4724): 141 • Time Elapse Update— Update the value of each particle according to the transition model. For a particle in stateti, sample the updated value from the probability distribution given byP (Ti+1| ti...

## What is d-separation in a Bayes net?

Expected pages 100-111 | retrieval precision: 3/3 = 1.00

**Answer** (55 s):

D-separation in a Bayes Net refers to the property of the structure of the graph that implies conditional independence between two nodes X and Y, given a set of observed variables {Z1, ..., Zk}. [1]

**Sources:**

- [1] book_1.pdf, page 105, chunk 1 (similarity 0.6592): the following statement be true:X⊥ ⊥Y|{Z1, . . . Zk}? D-separation (directed separation) is a property of the structure of the Bayes Net graph that implies this conditional independence relationship, ...
- [2] book_1.pdf, page 105, chunk 0 (similarity 0.6414): 105 Common Effect can be viewed as “opposite” to Causal Chains and Common Cause —X and Z are guaranteed to be independent ifY is not conditioned on. But when conditioned onY, X and Z may be dependent ...
- [3] book_1.pdf, page 100, chunk 0 (similarity 0.6295): 100 6.5 D-Separation 6.5 D-Separation One useful question to ask about a set of random variables is whether or not one variable is independent from another, or if one random variable is conditionally ...

## Who won the FIFA World Cup in 2018?

Out-of-scope question (answer is not in the book)

**Answer** (36 s):

I could not find the answer in the books.

**Sources:**

- [1] book_1.pdf, page 45, chunk 1 (similarity 0.1638): or may not bezero-sum. The first class of games we’ll cover aredeterministic zero-sum games , where actions are deterministic and our gain is directly equivalent to our opponent’s loss and vice versa....
- [2] book_1.pdf, page 46, chunk 0 (similarity 0.1573): 46 Figure 1: Common Games...
- [3] book_1.pdf, page 45, chunk 2 (similarity 0.1465): , Deep Blue became the first computer agent to defeat human chess champion Garry Kasparov in a six-game match. Deep Blue was constructed to use extremely sophisticated methods to evaluate over 200 mil...
