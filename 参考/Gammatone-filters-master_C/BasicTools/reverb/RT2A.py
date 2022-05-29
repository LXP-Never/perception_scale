import numpy as np


def RT2A(RT60, room_size, F_abs=None, c=343, A_air=None,
         estimator='Norris_Eyring'):
    """reverberation time to absorption coefficiences
    Args:
        RT60: reverberation time
        room_size: 3d room size
        F_abs: frequency of each absorption coefficience
        c: sound speed, default to 343m/s
        A_air:
        estimator:
    """
    if np.max(RT60) < 1e-10:
        return np.ones((6, 6), dtype=np.float32)

    if F_abs is None:
        F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])

    if A_air is None:
        humidity = 50
        A_air = 5.5e-4 * (50 / humidity) * ((F_abs / 1000) ** 1.7)

    V_room = np.prod(room_size)  # Volume of room m^3
    S_wall_all = [room_size[0] * room_size[2],
                  room_size[1] * room_size[2],
                  room_size[0] * room_size[1]]
    S_room = 2. * np.sum(S_wall_all)  # Total area of shoebox room surfaces

    if estimator == 'Sabine':
        A = np.divide(55.25/c*V_room/S_room, RT60)
    if estimator == 'SabineAir':
        A = (np.divide(55.25/c*V_room, RT60)-4*A_air*V_room)/S_room
    if estimator == 'SabineAirHiAbs':
        A = np.sqrt(2*(np.divide(55.25/c*V_room, RT60) - 4*A_air*V_room)+1)-1
    if estimator == 'Norris_Eyring':
        A = 1-np.exp((4*A_air*V_room-np.divide(55.25/c*V_room, RT60))/S_room)
    else:
        A = np.ones(6) * np.Inf
    return np.repeat(A.reshape([1, 6]), 6, axis=0)


def test_RT2A():
    from A2RT import A2RT
    F_abs = np.asarray([125, 250, 500, 1000, 2000, 4000])
    A_RT0_5 = np.asarray([0.2136, 0.2135, 0.2132, 0.2123, 0.2094, 0.1999])
    room_size = (5.1, 7.1, 3)
    A = np.repeat(A_RT0_5.reshape((-1, 1)), repeats=6, axis=1)
    RT = A2RT(room_size=room_size, A=A, F_abs=F_abs)
    print(RT)
    A = RT2A(RT60=RT, room_size=room_size, F_abs=F_abs)
    print(A)
