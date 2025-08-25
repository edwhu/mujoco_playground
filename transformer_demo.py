#!/usr/bin/env python3
"""
Minimal demonstration of tokenization + transformer encoder for touch sensors.
This shows how to convert the current concatenated observation format into tokens
and process them through a simple transformer encoder.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any
import numpy as np


class MultiModalityTokenEmbedding(nn.Module):
    """Separate embeddings for different modalities (touch, joint, action)."""
    embed_dim: int = 64
    
    def setup(self):
        # Separate encoders for each modality
        self.touch_encoder = nn.Dense(self.embed_dim, name="touch_encoder")
        self.joint_encoder = nn.Dense(self.embed_dim, name="joint_encoder") 
        self.action_encoder = nn.Dense(self.embed_dim, name="action_encoder")
        
    def __call__(self, touch_values, joint_values, action_values):
        """
        Encode each modality separately with specialized encoders.
        
        Args:
            touch_values: (batch_size, 20) - touch sensor values
            joint_values: (batch_size, 16) - joint angle values
            action_values: (batch_size, 16) - action values
            
        Returns:
            Dict containing embedded values for each modality
        """
        # Add feature dimension for each modality and encode
        touch_expanded = jnp.expand_dims(touch_values, -1)  # (batch_size, 20, 1)
        joint_expanded = jnp.expand_dims(joint_values, -1)  # (batch_size, 16, 1)
        action_expanded = jnp.expand_dims(action_values, -1)  # (batch_size, 16, 1)
        
        touch_embeds = self.touch_encoder(touch_expanded)  # (batch_size, 20, embed_dim)
        joint_embeds = self.joint_encoder(joint_expanded)  # (batch_size, 16, embed_dim) 
        action_embeds = self.action_encoder(action_expanded)  # (batch_size, 16, embed_dim)
        
        return {
            'touch': touch_embeds,
            'joint': joint_embeds, 
            'action': action_embeds
        }


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings for different token positions."""
    max_tokens: int
    embed_dim: int
    
    def setup(self):
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.max_tokens, self.embed_dim)
        )
    
    def __call__(self, num_tokens):
        return self.pos_embedding[:num_tokens]


class ModalityEmbedding(nn.Module):
    """Learned embeddings to distinguish different input modalities."""
    num_modalities: int = 3  # touch, joint, action
    embed_dim: int = 64
    
    def setup(self):
        self.modality_embedding = self.param(
            'modality_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.num_modalities, self.embed_dim)
        )
    
    def __call__(self, modality_ids):
        # modality_ids: (num_tokens,) - integer IDs for each token's modality
        return self.modality_embedding[modality_ids]


class SimpleTransformerBlock(nn.Module):
    """Single transformer encoder block with multi-head attention."""
    embed_dim: int = 64
    num_heads: int = 4
    mlp_dim: int = 128
    dropout_rate: float = 0.1
    
    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        # Correctly define the MLP layers without the Sequential wrapper for now
        # to handle the deterministic flag properly.
        self.dense1 = nn.Dense(self.mlp_dim)
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.dense2 = nn.Dense(self.embed_dim)
        self.dropout2 = nn.Dropout(rate=self.dropout_rate)
        
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
    
    def __call__(self, x, attention_mask=None, training=True):
        # Multi-head self-attention with residual connection
        # The mask is used here to prevent attention to certain tokens
        attn_output = self.attention(x, x, mask=attention_mask, deterministic=not training)
        x = self.layer_norm1(x + attn_output)
        
        # MLP with residual connection
        y = self.dense1(x)
        y = nn.gelu(y)
        y = self.dropout1(y, deterministic=not training)
        y = self.dense2(y)
        y = self.dropout2(y, deterministic=not training)
        
        x = self.layer_norm2(x + y)
        
        return x


class TouchTransformerEncoder(nn.Module):
    """Complete transformer encoder for processing touch + proprioceptive tokens with separate modality encoders."""
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    max_tokens: int = 60  # 20 touch + 16 joint + 16 action + some buffer
    
    def setup(self):
        # Multi-modality token embedding with separate encoders
        self.token_embedding = MultiModalityTokenEmbedding(self.embed_dim)
        self.pos_embedding = PositionalEmbedding(self.max_tokens, self.embed_dim)
        self.modality_embedding = ModalityEmbedding(embed_dim=self.embed_dim)
        
        self.transformer_blocks = [
            SimpleTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads
            ) for _ in range(self.num_layers)
        ]
        
        # Output projection to get final representation
        self.output_proj = nn.Dense(128)  # Project to desired output size
    
    def __call__(self, tokens_dict, training=True):
        """
        Process tokenized inputs through transformer encoder using separate modality encoders.
        
        Args:
            tokens_dict: Dict with keys:
                - touch_values: (batch_size, 20) - touch sensor values
                - joint_values: (batch_size, 16) - joint angle values  
                - action_values: (batch_size, 16) - action values
                - modality_ids: (num_tokens,) - modality ID for each token
                - num_tokens: int - total number of tokens (52)
                - attention_mask: (batch_size, num_tokens) - Optional boolean mask
        """
        touch_values = tokens_dict['touch_values']
        joint_values = tokens_dict['joint_values'] 
        action_values = tokens_dict['action_values']
        modality_ids = tokens_dict['modality_ids']
        num_tokens = tokens_dict['num_tokens']
        attention_mask = tokens_dict.get('attention_mask') # (batch_size, num_tokens)
        batch_size = touch_values.shape[0]
        
        # Reshape mask for broadcasting. Flax's MultiHeadDotProductAttention expects
        # a mask shape of (batch_size, 1, 1, seq_len) to correctly broadcast
        # across the heads and query dimensions.
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2)) # (batch, 1, 1, seq_len)
        
        # Get separate embeddings for each modality using specialized encoders
        modality_embeds = self.token_embedding(touch_values, joint_values, action_values)
        
        # Concatenate the embeddings back into sequence format for transformer processing
        token_embeds = jnp.concatenate([
            modality_embeds['touch'],   # (batch_size, 20, embed_dim)
            modality_embeds['joint'],   # (batch_size, 16, embed_dim) 
            modality_embeds['action']   # (batch_size, 16, embed_dim)
        ], axis=1)  # (batch_size, 52, embed_dim)
        
        # Add positional embeddings (broadcast across batch)
        pos_embeds = self.pos_embedding(num_tokens)  # (52, embed_dim)
        pos_embeds = jnp.expand_dims(pos_embeds, 0)  # (1, 52, embed_dim)
        
        # Add modality embeddings (broadcast across batch)  
        mod_embeds = self.modality_embedding(modality_ids)  # (52, embed_dim)
        mod_embeds = jnp.expand_dims(mod_embeds, 0)  # (1, 52, embed_dim)
        
        # Combine all embeddings
        x = token_embeds + pos_embeds + mod_embeds  # (batch_size, 52, embed_dim)
        
        # Pass through transformer blocks, applying the attention mask
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask, training=training)
        
        # Global average pooling to get single representation
        pooled = jnp.mean(x, axis=1)  # (batch_size, embed_dim)
        
        # Final projection
        output = self.output_proj(pooled)  # (batch_size, 128)
        
        return output, x  # Return both pooled output and token representations


def tokenize_observations(
    touch_sensors, joint_angles, last_actions, missing_touch_indices=None
):
    """
    Convert the current concatenated observation format into separate modality tokens.
    Also creates an attention mask if some sensors are missing.
    
    Args:
        touch_sensors: (batch_size, 20) - binary touch sensor values
        joint_angles: (batch_size, 16) - noisy joint angle values  
        last_actions: (batch_size, 16) - previous action values
        missing_touch_indices: Optional[List[int]] - a list of indices for unavailable
            touch sensors. These will be masked out in the attention mechanism.
    
    Returns:
        Dict with separate arrays for each modality and combined metadata
    """
    batch_size = touch_sensors.shape[0]
    num_tokens = 52

    # Keep modalities separate for specialized encoding
    modality_data = {
        'touch_values': touch_sensors,     # (batch_size, 20)
        'joint_values': joint_angles,      # (batch_size, 16) 
        'action_values': last_actions      # (batch_size, 16)
    }
    
    # Create modality IDs for positional embeddings (after concatenation)
    modality_ids = jnp.concatenate([
        jnp.zeros(20, dtype=jnp.int32),  # Touch sensors = modality 0
        jnp.ones(16, dtype=jnp.int32),   # Joint angles = modality 1  
        jnp.full(16, 2, dtype=jnp.int32) # Actions = modality 2
    ])
    
    # Also provide concatenated version for positional embeddings
    all_values = jnp.concatenate([touch_sensors, joint_angles, last_actions], axis=1)
    
    # Create attention mask. True = keep, False = mask out.
    attention_mask = jnp.ones((batch_size, num_tokens), dtype=jnp.bool_)
    if missing_touch_indices is not None:
        # Since touch sensors are the first tokens, their indices are direct.
        attention_mask = attention_mask.at[:, missing_touch_indices].set(False)

    return {
        **modality_data,
        'modality_ids': modality_ids,
        'all_values': all_values,  # Still needed for some operations
        'attention_mask': attention_mask,
        'num_tokens': num_tokens,
    }


def demo_transformer_encoder():
    """Demonstrate the complete tokenization + transformer pipeline."""
    print("=" * 60)
    print("TRANSFORMER ENCODER DEMO FOR TOUCH SENSORS")
    print("=" * 60)
    
    # Create some random data mimicking the current environment observations
    batch_size = 4
    rng = jax.random.PRNGKey(42)
    
    # Generate random observations (similar to current env format)
    rng, touch_rng, joint_rng, action_rng = jax.random.split(rng, 4)
    
    # Touch sensors: binary values (0 or 1)
    touch_sensors = jax.random.bernoulli(touch_rng, 0.3, (batch_size, 20)).astype(jnp.float32)
    
    # Joint angles: continuous values in reasonable range
    joint_angles = jax.random.normal(joint_rng, (batch_size, 16)) * 0.5
    
    # Last actions: continuous values 
    last_actions = jax.random.normal(action_rng, (batch_size, 16)) * 0.3
    
    print(f"\nInput Data Shapes:")
    print(f"  Touch sensors: {touch_sensors.shape} (binary)")
    print(f"  Joint angles: {joint_angles.shape} (continuous)")
    print(f"  Last actions: {last_actions.shape} (continuous)")
    
    # Tokenize the observations
    tokens = tokenize_observations(touch_sensors, joint_angles, last_actions)
    
    print(f"\nTokenized Data:")
    print(f"  Touch values shape: {tokens['touch_values'].shape}")
    print(f"  Joint values shape: {tokens['joint_values'].shape}")
    print(f"  Action values shape: {tokens['action_values'].shape}")
    print(f"  Modality IDs shape: {tokens['modality_ids'].shape}")
    print(f"  Modality IDs: {tokens['modality_ids']}")
    print(f"    0 = touch, 1 = joint, 2 = action")
    print(f"  Total tokens: {tokens['num_tokens']}")
    
    # Initialize transformer encoder
    model = TouchTransformerEncoder(
        embed_dim=64,
        num_heads=4,
        num_layers=1
    )
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, tokens, training=False)
    
    print(f"\nTransformer Architecture:")
    print(f"  Embedding dimension: 64")
    print(f"  Number of attention heads: 4") 
    print(f"  Number of transformer layers: 1")
    print(f"  Max tokens supported: 60")
    print(f"  Separate encoders: Touch, Joint, Action (NEW!)")
    
    # Forward pass
    output, token_representations = model.apply(params, tokens, training=False)
    
    print(f"\nTransformer Output:")
    print(f"  Pooled output shape: {output.shape}")
    print(f"  Token representations shape: {token_representations.shape}")
    print(f"  Output values (first sample): {output[0, :8]}")  # Show first 8 values
    
    # Demonstrate attention to specific modalities
    print(f"\nToken Representation Analysis:")
    print(f"  Touch token representations: {token_representations[0, :3, :3]}")  # First 3 touch tokens, first 3 dims
    print(f"  Joint token representations: {token_representations[0, 20:23, :3]}")  # First 3 joint tokens
    print(f"  Action token representations: {token_representations[0, 36:39, :3]}")  # First 3 action tokens
    
    # Show how you could mask specific sensors using an attention mask
    print(f"\nDemonstrating Attention Masking:")
    
    # Define which touch sensor indices are "missing"
    missing_indices = [5, 6, 7, 8, 9]
    print(f"  Missing touch sensor indices: {missing_indices}")
    
    # Create a new token dictionary with an attention mask
    # Note: We DO NOT change the underlying touch_sensor values.
    # The value 0.0 is meaningful (no force), and the mask handles "missing".
    masked_tokens = tokenize_observations(
        touch_sensors, joint_angles, last_actions, missing_touch_indices=missing_indices
    )
    
    print(f"  Shape of attention mask: {masked_tokens['attention_mask'].shape}")
    print(f"  Mask for first sample (first 15 tokens): {masked_tokens['attention_mask'][0, :15]}")
    
    # The model's __call__ function will automatically use the 'attention_mask'
    masked_output, _ = model.apply(params, masked_tokens, training=False)
    
    print(f"\n  Original output norm (all sensors visible): {jnp.linalg.norm(output[0]):.4f}")
    print(f"  Masked output norm (some sensors ignored): {jnp.linalg.norm(masked_output[0]):.4f}")
    print(f"  Output difference due to masking: {jnp.linalg.norm(output[0] - masked_output[0]):.4f}")
    
    print(f"\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    return model, params, tokens, output


if __name__ == "__main__":
    # Run the demonstration
    model, params, tokens, output = demo_transformer_encoder()
    
    print(f"\nKey Improvements with Multi-Encoder Architecture:")
    print(f"1. Separate specialized encoders for each modality:")
    print(f"   - Touch encoder: optimized for binary/sparse touch signals")  
    print(f"   - Joint encoder: optimized for continuous angle values")
    print(f"   - Action encoder: optimized for motor command representations")
    print(f"2. Each encoder can learn modality-specific transformations")
    print(f"3. More parameters but better specialization")
    
    print(f"\nNext Steps for Integration:")
    print(f"1. Modify PPO network factory to use TouchTransformerEncoder")
    print(f"2. Update environment observation format to support tokenization")
    print(f"3. Adjust training hyperparameters for transformer architecture")
    print(f"4. Implement attention visualization for research analysis")
