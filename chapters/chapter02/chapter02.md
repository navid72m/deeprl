# Chapter 2: Core Concepts - Key Notes

## Fundamental RL Components

### Core Concepts
- **Agent**: The decision-maker that learns and takes actions
- **Environment**: The world in which the agent operates and receives feedback
- **Action (A)**: Choices available to the agent at each step
- **Policy (π)**: Strategy/function that maps states to actions (the agent's behavior)
- **State (S)**: Current situation or configuration of the environment
- **Reward (R)**: Immediate feedback signal from the environment
- **Value (V)**: Expected cumulative reward from a given state (long-term perspective)

### The Agent-Environment Loop

*[rl-diagram.png]*

The fundamental cycle of reinforcement learning:
1. Agent observes current state `s_t`
2. Agent selects action `a_t` based on policy `π(a|s)`
3. Environment transitions to new state `s_{t+1}` and provides reward `r_{t+1}`
4. Process repeats with agent learning from experience

## Sequential Decision Problems

**Objective**: Find a sequence of decisions that maximizes cumulative reward over time

### Examples of Sequential Decision Problems:
- **Grid worlds**: Navigate through a grid to reach goal states
- **Mazes and box puzzles**: Find optimal paths while avoiding obstacles
- **Game playing**: Make moves that lead to winning outcomes

### RL Paradigm
The reinforcement learning cycle: Agent takes Action → Environment responds → Agent learns from feedback → Repeat

## Markov Decision Process (MDP)

### Markov Property
**Key Insight**: The future depends only on the present, not the past
- Next state depends solely on current state and chosen action
- No need to remember historical states or external information
- Mathematically: 

```math
P(S_{t+1} | S_t, A_t, S_{t-1}, \ldots, S_0) = P(S_{t+1} | S_t, A_t)
```

### MDP 5-Tuple Definition
An MDP is formally defined as: 

```math
\langle S, A, T, R, \gamma \rangle
```

1. **S**: Set of all possible states
2. **A**: Set of all possible actions  
3. **T**: Transition function - `P(s'|s,a)` = probability of reaching state s' from state s taking action a
4. **R**: Reward function - `R(s,a,s')` = immediate reward for transition
5. **γ (Gamma)**: Discount factor `(0 ≤ γ ≤ 1)`
   - Controls importance of future vs. immediate rewards
   - `γ = 0`: Only immediate rewards matter
   - `γ = 1`: All future rewards equally important
   - `γ < 1`: Future rewards are discounted

## Value Functions & Bellman Equations

### State Value Function
**\(V^{\pi}(s)\)**: Expected cumulative discounted reward starting from state \(s\) following policy \(\pi\)

\[
V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]
\]

### Action Value Function (Q-Function)
**\(Q^{\pi}(s,a)\)**: Expected cumulative discounted reward starting from state \(s\), taking action \(a\), then following policy \(\pi\)

\[
Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]
\]

### Bellman Equations

#### Bellman Equation for State Values
\[
V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^{\pi}(s')\right]
\]

**Intuition**: Value of current state = immediate reward + discounted value of next states

#### Bellman Equation for Action Values  
\[
Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')Q^{\pi}(s',a')\right]
\]

#### Bellman Optimality Equations
For optimal policy \(\pi^*\):
\[
V^*(s) = \max_{a} \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]
\]

\[
Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]
\]

## Traces and Eligibility

### Eligibility Traces
**Purpose**: Bridge between temporal difference (TD) and Monte Carlo methods
- Assign credit to states/actions that led to rewards
- Decay over time based on recency and frequency of visits

### Trace Types
1. **Accumulating Traces**: 
```math
e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbf{1}(S_t = s)
```

2. **Replacing Traces**: 
```math
e_t(s) = \begin{cases} 
\gamma\lambda e_{t-1}(s) & \text{if } S_t \neq s \\ 
1 & \text{if } S_t = s 
\end{cases}
```

Where:
- \(\lambda\) (lambda): Trace decay parameter `(0 ≤ λ ≤ 1)`
- **1**(S_t = s): Indicator function (1 if true, 0 otherwise)

### Benefits of Traces
- **Faster learning**: Credit assignment happens immediately
- **Better sample efficiency**: One experience updates multiple states
- **Smooths learning**: Reduces variance in value estimates

## Key Mathematical Relationships

### Return (Cumulative Reward)
\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\]

### Policy Relationship
\[
V^{\pi}(s) = \sum_{a} \pi(a|s) Q^{\pi}(s,a)
\]

\[
Q^{\pi}(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^{\pi}(s')\right]
\]

## Important Insights

1. **Markov Property** enables efficient computation - we don't need infinite memory
2. **Discount factor** balances immediate vs. future rewards and ensures convergence
3. **Bellman equations** provide recursive relationships that enable dynamic programming
4. **Value functions** capture the long-term desirability of states and actions
5. **Traces** provide a mechanism for efficient credit assignment across time

## Questions & Reflections

- How does the discount factor \(\gamma\) affect learning behavior in different environments?
- When might the Markov assumption be violated in real-world problems?
- How do eligibility traces change the learning dynamics compared to basic TD methods?
- What's the relationship between policy evaluation and the Bellman equation?