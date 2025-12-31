# AP Calculus BC AI Mastermind

An intelligent AI-powered chatbot specialized in AP Calculus BC, providing step-by-step explanations, LaTeX-rendered mathematics, and integrated graphing capabilities using OpenAI's GPT-4o-mini.

## Features

- **Expert Tutoring**: Get detailed, step-by-step solutions to AP Calculus BC problems
- **LaTeX Rendering**: All mathematical notation is beautifully rendered using Streamlit's native KaTeX support
- **Unit-Specific Focus**: Choose from 5 specialized units:
  - Integration Techniques (Parts, Partial Fractions, Improper)
  - Differential Equations (Euler's Method, Logistic Growth)
  - Parametrics & Vectors (Position, Velocity, Arc Length)
  - Polar Curves (Area, Slopes)
  - Infinite Series (Ratio Test, Taylor/Maclaurin, Lagrange Error)
- **Interactive Graphing**: Visualize functions using Matplotlib with SymPy parsing
- **Conversation History**: Maintains context throughout your session
- **Clear Interface**: Modern Streamlit UI with sidebar controls

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```
   - **Important**: Never commit your `.env` file to version control (it's already in `.gitignore`)

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select a Unit Focus** (optional): Use the sidebar to choose a specific AP Calculus BC unit for more targeted assistance

2. **Ask Questions**: Type your calculus question in the chat input at the bottom

3. **View Solutions**: The AI will provide step-by-step solutions with properly formatted LaTeX mathematics

4. **Graph Functions**: When discussing functions, the AI will offer to graph them. Click the graph button to visualize

5. **Clear Chat**: Use the "Clear Chat" button in the sidebar to start a new conversation

## Example Usage Scenarios

### Integration by Parts
```
User: "How do I integrate x*e^x using integration by parts?"
AI: Provides step-by-step solution with LaTeX, explains LIATE rule, offers to graph
```

### Series Convergence
```
User: "Does the series sum of n!/n^n converge?"
AI: Applies Ratio Test, shows limit calculations step-by-step, explains reasoning
```

### Parametric Curves
```
User: "Find the arc length of x(t)=t^2, y(t)=t^3 from t=0 to t=2"
AI: Shows arc length formula, sets up integral, solves step-by-step
```

### Polar Curves
```
User: "Find the area inside r=2+2cos(theta)"
AI: Explains polar area formula, sets up integral, shows evaluation
```

### Differential Equations
```
User: "Solve dy/dx = y(1-y) with y(0)=0.5"
AI: Recognizes logistic growth, separates variables, solves with initial condition
```

## Project Structure

```
APGPT/
├── .env                    # Your OpenAI API key (create this)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── app.py                  # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── openai_handler.py   # OpenAI API integration
│   ├── graph_engine.py     # Matplotlib plotting with SymPy
│   └── prompts.py          # Unit-specific system prompts
└── README.md               # This file
```

## Dependencies

- `streamlit` - Web interface framework
- `openai` - GPT-4o-mini API client
- `python-dotenv` - Environment variable management
- `matplotlib` - Graphing and visualization
- `numpy` - Numerical computations
- `sympy` - Symbolic mathematics parsing

## Notes

- **API Costs**: This application uses GPT-4o-mini, which incurs API costs. Monitor your usage on the OpenAI platform
- **Internet Required**: The application requires an active internet connection to communicate with OpenAI's API
- **Graphing Limitations**: The graphing engine works best with standard mathematical functions. Complex parametric or polar equations may require manual input

## Troubleshooting

**"API Key Invalid" error**:
- Ensure your `.env` file exists in the project root
- Verify the API key is correct (no extra spaces or quotes)
- Check that your OpenAI account has available credits

**Graphing errors**:
- Try simplifying the function notation
- Use standard mathematical syntax (e.g., `x^2` instead of `x²`)
- For parametric/polar functions, specify them explicitly

**Import errors**:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8+

## License

This project is provided as-is for educational purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!

