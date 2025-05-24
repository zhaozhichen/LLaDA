import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time
import re

# Helper function to convert history to Gradio 'messages' format

def history_to_messages(history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
        
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    
    return constraints

def format_chat_history(history):
    """
    Format chat history for the LLaDA model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        constraints: Dictionary mapping positions to words
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    
    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Mark prompt positions to exclude them from masking during classifier-free guidance
    prompt_index = (x != MASK_ID)
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Track the current state of x for visualization
    current_x = x.clone()

    # Process each block
    for num_block in range(num_blocks):
        # Calculate the start and end indices for the current block
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        
        # Get mask indices for the current block
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        
        # Skip if no masks in this block
        if not block_mask_index.any():
            continue
        
        # Calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Process each step
        for i in range(steps_per_block):
            # Get all mask positions in the current sequence
            mask_index = (x == MASK_ID)
            
            # Skip if no masks
            if not mask_index.any():
                break
            
            # Apply classifier-free guidance if enabled
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            # Don't consider positions beyond the current block
            x0_p[:, block_end:] = -float('inf')
            
            # Apply predictions where we have masks
            old_x = x.clone()
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            
            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # Only consider positions within the current block for unmasking
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:  # Not the last step
                    # Take top-k confidences
                    _, select_indices = torch.topk(block_confidence, 
                                                  k=min(num_transfer_tokens[j, i].item(), 
                                                       block_confidence.numel()))
                    # Adjust indices to global positions
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:  # Last step - unmask everything remaining
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]
            
            # Apply the selected tokens
            x = torch.where(transfer_index, x0, x)
            
            # Ensure constraints are maintained
            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id
            
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
                    
                elif old_x[0, pos] == MASK_ID:
                    # Newly revealed in this step
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    # Color based on confidence
                    confidence = float(x0_p[0, pos].cpu())
                    if confidence < 0.3:
                        color = "#FF6666"  # Light red
                    elif confidence < 0.7:
                        color = "#FFAA33"  # Orange
                    else:
                        color = "#66CC66"  # Light green
                        
                    current_state.append((token, color))
                    
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, "#6699CC"))  # Light blue
            
            visualization_states.append(current_state)
    
    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text

css = '''
.category-legend{display:none}
button{height: 60px}
'''
def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# LLaDA - Large Language Diffusion Model Demo")
        gr.Markdown("[model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), [project page](https://ml-gsai.github.io/LLaDA-demo/)")
        
        # STATE MANAGEMENT
        chat_history = gr.State([])
        last_vis_states = gr.State([])
        last_response_text = gr.State("")
        
        # UI COMPONENTS
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500, type='messages')
                
                # Message input
                with gr.Group():
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Your Message", 
                            placeholder="Type your message here...",
                            show_label=False
                        )
                        send_btn = gr.Button("Send")
                
                constraints_input = gr.Textbox(
                    label="Word Constraints", 
                    info="This model allows for placing specific words at specific positions using 'position:word' format. Example: 1st word once, 6th word 'upon' and 11th word 'time', would be: '0:Once, 5:upon, 10:time",
                    placeholder="0:Once, 5:upon, 10:time",
                    value=""
                )
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
        
        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(
                    minimum=16, maximum=128, value=64, step=8,
                    label="Generation Length"
                )
                steps = gr.Slider(
                    minimum=8, maximum=64, value=32, step=4,
                    label="Denoising Steps"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                    label="Temperature"
                )
                cfg_scale = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="CFG Scale"
                )
            with gr.Row():
                block_length = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Block Length"
                )
                remasking_strategy = gr.Radio(
                    choices=["low_confidence", "random"],
                    value="random",
                    label="Remasking Strategy"
                )
            with gr.Row():
                # Replace visualization_delay slider with speed adjust bar
                speed_options = [
                    ("*0.25", 0.4),
                    ("*0.5", 0.2),
                    ("*1", 0.1),
                    ("*1.5", 0.066),
                    ("*2", 0.05)
                ]
                speed_labels = [opt[0] for opt in speed_options]
                speed_values = [opt[1] for opt in speed_options]
                speed_map = dict(speed_options)
                speed_select = gr.Radio(
                    choices=speed_labels,
                    value="*1",
                    label="Visualization Speed"
                )
        
        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        # Add Replay button
        replay_btn = gr.Button("Replay Denoising")
        
        # Add replay progress slider
        replay_progress = gr.Slider(
            minimum=0,
            maximum=1,  # Changed from 0 to 1 to ensure initial non-zero range
            value=0,
            step=1,
            label="Replay Progress",
            interactive=True,
            visible=True
        )
        current_replay_step = gr.State(0)
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history, gen_length, steps, constraints, speed_label):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_for_display = history.copy()
                debug_out = history_to_messages(history_for_display)
                print("DEBUG chatbot_ui output (user_message_submitted empty):", debug_out)
                return history, debug_out, "", [], "", [], ""
            # Add user message to history
            history = add_message(history, message, None)
            # Format for display - temporarily show user message with empty response
            history_for_display = history.copy()
            # Clear the input
            message_out = ""
            debug_out = history_to_messages(history_for_display)
            print("DEBUG chatbot_ui output (user_message_submitted):", debug_out)
            return history, debug_out, message_out, [], "", [], ""
            
        def bot_response(history, gen_length, steps, constraints, speed_label, temperature, cfg_scale, block_length, remasking):
            """Generate bot response for the latest message"""
            # history is the gr.State's value
            if not history:
                processed_chat_for_ui = history_to_messages([])
                print("DEBUG chatbot_ui output (bot_response empty):", processed_chat_for_ui)
                # chat_history, chatbot_ui, output_vis, current_response, last_vis_states, last_response_text
                return [], processed_chat_for_ui, [], "", [], ""
            
            last_user_message = history[-1][0]
            try:
                messages_for_model = format_chat_history(history[:-1])
                messages_for_model.append({"role": "user", "content": last_user_message})
                
                parsed_constraints = parse_constraints(constraints)
                speed_map = {"*0.25": 0.4, "*0.5": 0.2, "*1": 0.1, "*1.5": 0.066, "*2": 0.05}
                delay = speed_map.get(speed_label, 0.1)
                
                vis_states, response_text = generate_response_with_visualization(
                    model, tokenizer, device, 
                    messages_for_model, 
                    gen_length=gen_length, 
                    steps=steps,
                    constraints=parsed_constraints,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    block_length=block_length,
                    remasking=remasking
                )
                
                history[-1][1] = response_text # Modify the history state
                
                processed_chat_for_ui = history_to_messages(history)
                # Add debug prints here
                print(f"DEBUG bot_response inputs: gen_length={gen_length}, steps={steps}, block_length={block_length}, remasking={remasking}, temp={temperature}, cfg={cfg_scale}")
                print(f"DEBUG bot_response: len(vis_states) from generate_response_with_visualization = {len(vis_states)}")

                print("DEBUG chatbot_ui output (bot_response first yield):", processed_chat_for_ui)
                # chat_history (raw), chatbot_ui (processed), output_vis, current_response, last_vis_states, last_response_text
                yield history, processed_chat_for_ui, vis_states[0], response_text, vis_states, response_text
                
                for state_vis in vis_states[1:]: # Renamed 'state' to 'state_vis' to avoid conflict if 'history' becomes gr.State object
                    time.sleep(delay)
                    # history object is already updated, processed_chat_for_ui uses the updated history
                    processed_chat_for_ui = history_to_messages(history) 
                    print("DEBUG chatbot_ui output (bot_response yield):", processed_chat_for_ui)
                    yield history, processed_chat_for_ui, state_vis, response_text, vis_states, response_text
            except Exception as e:
                error_msg = str(e)
                print(f"Error in bot_response: {error_msg}")
                error_vis = [(error_msg, "red")]
                processed_chat_for_ui = history_to_messages(history) # Show current history even on error
                print("DEBUG chatbot_ui output (bot_response error):", processed_chat_for_ui)
                # chat_history, chatbot_ui, output_vis, current_response, last_vis_states, last_response_text
                yield history, processed_chat_for_ui, error_vis, error_msg, [], error_msg
        
        def clear_conversation():
            """Clear the conversation history"""
            # Prepare the correctly formatted empty history for the chatbot UI
            processed_empty_history_for_ui = history_to_messages([])
            print("DEBUG chatbot_ui output (clear_conversation):", processed_empty_history_for_ui)
            
            # chat_history (state), chatbot_ui, current_response, output_vis, last_vis_states, last_response_text
            return [], processed_empty_history_for_ui, "", [], [], ""
        
        def replay_denoising(vis_states, response_text, history, speed_label):
            speed_map = {"*0.25": 0.4, "*0.5": 0.2, "*1": 0.1, "*1.5": 0.066, "*2": 0.05}
            delay = speed_map.get(speed_label, 0.1)
            debug_out_history = history_to_messages(history) # Prepare history for chatbot once

            if not vis_states:
                print("DEBUG chatbot_ui output (replay_denoising empty):", debug_out_history)
                # chatbot_ui, output_vis, current_response, replay_progress (value update), replay_progress (max update)
                return debug_out_history, [], response_text, gr.update(value=0), gr.update(maximum=1) # Keep a valid range
            
            max_step = len(vis_states) - 1
            for i, state_vis in enumerate(vis_states):
                time.sleep(delay)
                print(f"DEBUG chatbot_ui output (replay_denoising step {i}):", debug_out_history)
                # chatbot_ui, output_vis, current_response, replay_progress (value + max update), replay_progress (value + max update)
                # We yield the same gr.update object to both slider outputs. Gradio should handle this.
                slider_update = gr.update(value=i, maximum=max_step)
                yield debug_out_history, state_vis, response_text, slider_update, slider_update

        def set_replay_step(vis_states, response_text, history, step):
            processed_history_for_ui = history_to_messages(history)
            if not vis_states or step is None or not isinstance(step, int) or step < 0 or step >= len(vis_states):
                print("DEBUG chatbot_ui output (set_replay_step empty or invalid step):", processed_history_for_ui)
                # chatbot_ui, output_vis, current_response
                return processed_history_for_ui, [], response_text 
            
            current_vis_frame = vis_states[step]
            print(f"DEBUG chatbot_ui output (set_replay_step to {step}):", processed_history_for_ui)
            # chatbot_ui, output_vis, current_response
            return processed_history_for_ui, current_vis_frame, response_text
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history, chatbot_ui, current_response, output_vis, last_vis_states, last_response_text]
        )
        
        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history, gen_length, steps, constraints_input, speed_select],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, current_response, last_vis_states, last_response_text]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history, gen_length, steps, constraints_input, speed_select],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, current_response, last_vis_states, last_response_text]
        )
        
        # Step 2: Generate bot response
        # This happens after the user message is displayed
        msg_submit.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length, steps, constraints_input, 
                speed_select, temperature, cfg_scale, block_length,
                remasking_strategy
            ],
            outputs=[chat_history, chatbot_ui, output_vis, current_response, last_vis_states, last_response_text]
        )
        
        send_click.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length, steps, constraints_input, 
                speed_select, temperature, cfg_scale, block_length,
                remasking_strategy
            ],
            outputs=[chat_history, chatbot_ui, output_vis, current_response, last_vis_states, last_response_text]
        )
        
        # Connect replay button
        replay_btn.click(
            fn=replay_denoising,
            inputs=[last_vis_states, last_response_text, chat_history, speed_select],
            outputs=[chatbot_ui, output_vis, current_response, replay_progress, replay_progress]
        )
        
        # Connect slider to set frame
        replay_progress.change(
            fn=set_replay_step,
            inputs=[last_vis_states, last_response_text, chat_history, replay_progress],
            outputs=[chatbot_ui, output_vis, current_response]
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(share=True)
