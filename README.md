# Connect 4 Web Game

A full-stack web-based Connect 4 game built with Django and JavaScript, featuring an AI opponent powered by a minimax algorithm.

## Features

- **Interactive Gameplay**: Play against an AI opponent on a 6×7 board
- **AI Engine**: Minimax algorithm with 4-move lookahead depth
- **Strategic Scoring**: AI evaluates 5 heuristic factors including center preference and threat detection
- **Real-time Updates**: Responsive JavaScript frontend with instant board rendering
- **Session Management**: Game state persisted using Django sessions

## Tech Stack

- **Backend**: Django 5.2, Python 3.x, NumPy
- **Frontend**: Vanilla JavaScript, HTML5
- **Database**: SQLite3
- **Game Logic**: NumPy-based board representation

## Project Structure

```
connect4_web/
├── connect4_web/          # Django project configuration
│   ├── settings.py        # Project settings
│   ├── urls.py            # URL routing
│   ├── wsgi.py            # WSGI configuration
│   └── asgi.py            # ASGI configuration
├── game/                  # Game app
│   ├── models.py          # Data models
│   ├── views.py           # API endpoints
│   ├── urls.py            # Game routes
│   ├── Connect_4.py       # Core game logic (6×7 board, win detection, piece placement)
│   ├── agent.py           # AI opponent (minimax algorithm, position scoring)
│   └── templates/
│       └── game/
│           └── index.html # Interactive game UI
├── manage.py              # Django management script
├── db.sqlite3             # SQLite database
└── requirements.txt       # Python dependencies
```

## Game Rules

- Standard Connect 4 rules: Get 4 pieces in a row (horizontal, vertical, or diagonal) to win
- Players alternate turns (human vs AI)
- Board: 6 rows × 7 columns = 42 positions
- Pieces drop to the lowest available position in selected column

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aayu62/Connect-4-game.git
   cd Connect-4-game
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run migrations** (if needed):
   ```bash
   python manage.py migrate
   ```

4. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

5. **Play the game**:
   - Open your browser to `http://localhost:8000`
   - Click on columns to place your pieces (🔴 = Player, 🟡 = AI)
   - Click "Restart Game" to start a new game

## API Endpoints

- `GET /` - Home page with game board
- `GET /start/` - Initialize a new game and return empty board
- `POST /move/` - Submit player move and get AI response
- `POST /restart/` - Reset game state and return empty board

## AI Algorithm

The AI opponent uses the **minimax algorithm** with:
- **Depth**: 4-move lookahead (balances performance vs. decision quality)
- **Heuristics**: Evaluates 5 factors per position
  - Center column preference (strategic positioning)
  - Horizontal threat/opportunity patterns
  - Vertical threat/opportunity patterns
  - Diagonal threats/opportunity patterns (2 directions)
- **Scoring**: Maximizes own score while minimizing opponent's score

## Example Request/Response

**Start Game**:
```
GET /start/
Response: { "message": "Game started", "board": [[0,0,0,...]] }
```

**Make Move**:
```
POST /move/
Body: { "col": 3 }
Response: { 
  "board": [[0,0,0,1,...]], 
  "agent_move": 5,
  "winner": null 
}
```

## Performance Notes

- Board state evaluation: O(n) where n = board size (42)
- Minimax complexity: ~O(7^4) ≈ 2,400 positions evaluated per turn
- Average AI response time: <100ms on modern hardware

## Future Enhancements

- [ ] Draw condition detection (board full)
- [ ] Game history/move replay
- [ ] Difficulty levels (adjustable minimax depth)
- [ ] Multiplayer support (socket.io)
- [ ] Enhanced UI/CSS styling
- [ ] Unit/integration tests
- [ ] Leaderboard system

## License

This project is open source.

## Author

Aayu62
