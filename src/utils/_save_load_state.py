import pickle


def save_state(path: str, session_state: dict) -> None:
    pickle_data = {
        'session_state': {key: value for key, value in session_state.items()},
    }
    with open(path, 'wb') as f: 
        pickle.dump(pickle_data, f)


def load_state(path: str, session_state):
    """return: st.SessionStateProxy"""
    try:
        with open(path, 'rb') as f:
            session_state_dict = pickle.load(f)['session_state']
    except FileNotFoundError:
        return session_state

    for key, value in session_state_dict.items():
        session_state[key] = value
    return session_state
