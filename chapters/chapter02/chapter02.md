
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

![The Agent-Environment Loop](rl-diagram.png)

The fundamental cycle of reinforcement learning:
1. Agent observes current state `sₜ`  
2. Agent selects action `aₜ` based on policy `π(a|s)`  
3. Environment transitions to new state `sₜ₊₁` and provides reward `rₜ₊₁`  
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
Mathematically:  
`P(Sₜ₊₁ | Sₜ, Aₜ, Sₜ₋₁, ..., S₀) = P(Sₜ₊₁ | Sₜ, Aₜ)`

### MDP 5-Tuple Definition
An MDP is formally defined as:  
`⟨S, A, T, R, γ⟩`

Where:
1. **S**: Set of all possible states  
2. **A**: Set of all possible actions  
3. **T**: Transition function — `P(s' | s, a)` = probability of reaching state `s'` from state `s` taking action `a`  
4. **R**: Reward function — `R(s, a, s')` = immediate reward for transition  
5. **γ (Gamma)**: Discount factor `(0 ≤ γ ≤ 1)`  
   - `γ = 0`: Only immediate rewards matter  
   - `γ = 1`: All future rewards equally important  
   - `γ < 1`: Future rewards are discounted  

## Value Functions & Bellman Equations

### State Value Function
`V^π(s)` is the expected cumulative discounted reward starting from state `s` following policy `π`:

```
V^π(s) = E_π[Gₜ | Sₜ = s] 
       = E_π[ ∑ₖ=₀^∞ γᵏ Rₜ₊ₖ₊₁ | Sₜ = s ]
```

### Action Value Function (Q-Function)
`Q^π(s, a)` is the expected return starting from state `s`, taking action `a`, and thereafter following policy `π`:

```
Q^π(s, a) = E_π[Gₜ | Sₜ = s, Aₜ = a]
```

### Bellman Equations

#### Bellman Equation for State Values
```
V^π(s) = ∑ₐ π(a|s) ∑ₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
```

#### Bellman Equation for Action Values
```
Q^π(s, a) = ∑ₛ' P(s'|s,a) [ R(s,a,s') + γ ∑ₐ' π(a'|s') Q^π(s', a') ]
```

#### Bellman Optimality Equations (for optimal policy π*)
```
V*(s) = maxₐ ∑ₛ' P(s'|s,a) [ R(s,a,s') + γ V*(s') ]
```

```
Q*(s, a) = ∑ₛ' P(s'|s,a) [ R(s,a,s') + γ maxₐ' Q*(s', a') ]
```

## Traces and Eligibility

### Eligibility Traces
Used to assign credit to states/actions based on recency and frequency.

#### Accumulating Traces:
```
eₜ(s) = γλ eₜ₋₁(s) + 1(Sₜ = s)
```

#### Replacing Traces:
```
eₜ(s) = {
  γλ eₜ₋₁(s),   if Sₜ ≠ s  
  1,            if Sₜ = s  
}
```

Where:
- `λ` is the trace decay parameter `(0 ≤ λ ≤ 1)`  
- `1(Sₜ = s)` is the indicator function (1 if true, 0 otherwise)

### Benefits of Traces
- Faster learning and credit assignment  
- One experience updates multiple states  
- Reduces variance and smooths learning  

## Key Mathematical Relationships

### Return (Cumulative Reward)
```
Gₜ = Rₜ₊₁ + γ Rₜ₊₂ + γ² Rₜ₊₃ + ... 
   = ∑ₖ=₀^∞ γᵏ Rₜ₊ₖ₊₁
```

### Policy Relationship
```
V^π(s) = ∑ₐ π(a|s) Q^π(s, a)
```

```
Q^π(s, a) = ∑ₛ' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
```

## Important Insights

1. **Markov Property** enables efficient computation without full history  
2. **Discount factor** γ balances short- vs. long-term rewards  
3. **Bellman equations** are the backbone of value estimation  
4. **Value functions** quantify long-term potential  
5. **Traces** improve credit assignment and learning dynamics  

## Questions & Reflections

- How does the discount factor `γ` affect learning behavior in different environments?  
- When might the Markov assumption be violated in real-world problems?  
- How do eligibility traces change the learning dynamics compared to basic TD methods?  
- What's the relationship between policy evaluation and the Bellman equation?
