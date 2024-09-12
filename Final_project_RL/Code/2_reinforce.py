"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.001

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    actor_path = "./saved_models/actor_0.pt"

    # ------------------------------------------
    # ---> TODO: UNCOMMENT FOR SECTION 4 ONLY
    env = CartpoleEnvV1()
    actor = ActorModelV1()
    actor_path = "./saved_models/actor_1.pt"
    # ------------------------------------------

    # Training mode
    actor.train()
    print(actor)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # ---> TODO: when do we stop the training?
    #keeping rolling average, using last 15 rewards
    ROLLING_WINDOW_SIZE = 15
    rolling_rewards = []

    #defining function which decides when to stop
    def should_stop_training(rolling_rewards, threshold):
        if len(rolling_rewards) >= ROLLING_WINDOW_SIZE:
            average_reward = sum(rolling_rewards) / len(rolling_rewards)
            #stopping when average reward is consistently more than threshold
            return average_reward >= threshold
        return False    

    # Run infinitely many episodes
    training_iteration = 0
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------

        # Current discounted reward
        discounted_reward = 0.0

        # List of all the discounted rewards, for each time step
        discounted_rewards = list()

        # ---> TODO: compute discounted rewards
        for reward in reversed(saved_rewards):
            discounted_reward *= DISCOUNT_FACTOR
            discounted_reward += reward
            #inserting in the beginning to bring rewards back in order
            discounted_rewards.insert(0, discounted_reward)

        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        # Update policy parameters
        # ------------------------------------------

        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, discounted_rewards):

            # ---> TODO: compute policy loss
            time_step_actor_loss = - g* torch.log(p)

            # Save it
            actor_loss.append(time_step_actor_loss)

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        rolling_rewards.append(episode_total_reward)
        
        # Maintain the size of rolling_rewards to be exactly ROLLING_WINDOW_SIZE
        if len(rolling_rewards) > ROLLING_WINDOW_SIZE:
            rolling_rewards.pop(0)  # Remove the oldest reward

        # Log results
        log_frequency = 50
        training_iteration += 1
        if training_iteration % log_frequency == 0:
            # Save neural network
            torch.save(actor, actor_path)

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))

            # ---> TODO: when do we stop the training?

        if should_stop_training(rolling_rewards, HORIZON): 
            print(f"Training has converged with rolling average: {sum(rolling_rewards) / len(rolling_rewards):.2f} in {training_iteration} iterations")
            break
