
import gym
import numpy as np
from scipy.integrate import ode
from .noise import absolute_noise, percent_noise


''' rate constants '''
K_POLY  = 0
K_ACT   = 1
K_DEACT = 2
K_TER   = 3

''' indices and actions '''
MONO  = 0
CU1   = 1
CU2   = 2
DORM1 = 3
SOL   = 4

''' chain types '''
DORM   = 'dorm'
RAD    = 'rad'
TER    = 'ter'
STABLE = 'dorm_ter'

''' slices for terminated chains '''
TER_SLICES = 'ter_slices'

''' epsilon for float comparison '''
EPS = 1e-3

''' scale ymax in rendering by this factor '''
MARGIN_SCALE = 1.1

''' ode integrator, integration method, and number of steps '''
ODE_INT    = 'vode'
ODE_METHOD = 'bdf'
ODE_NSTEPS = 5000


class ATRPBase(gym.Env):

    '''
    Length of `quant`:
        (3 + 2 * max_rad_len) if termination is off;
        (4 * max_rad_len + 2) if termination is on.

    Partition of `quant`:
        quant[0]           = mono  = [M]
        quant[1]           = cu_i  = [CuBr]
        quant[2]           = cu_ii = [CuBr2]
        quant[3:3+n]       = dorm  = [P1Br], ..., [PnBr]
        quant[3+n:3+2*n]   = rad   = [P1.], ..., [Pn.]
        quant[3+2*n:2+4*n] = ter   = [T2], ..., [T2n] (optional).

    About noises:
        There are a set of noise switches `XXX_noise` among input arguments.
        `None` indicates that there is no noise on XXX, otherwise it should be
        a 2-tuple with the format `(mode, scale)` where `mode` is in
        `['uniform', 'gaussian']` indicating the mode of noise. Explanations of
        `scale` can be found in the comments for individual noises.

    Simulation time related:
        step_time:       timestep between actions;
        step_time_noise: `scale` indicates percentage noise in `step_time`;
                         0.1 means +-10% maximum, etc.
        completion_time: final timestep to run the simulation to the end.

    Rate constants:
        k_prop:  rate constant for (monomer consumption);
        k_act:   rate constant for (dormant chain --> radical);
        k_deact: rate constant for (radical --> dormant chain);
        k_ter:   rate constant for (radical --> terminated chain);
        k_noise: `scale` indicates percentage noise in rate constants.

    Observation related:
        obs_noise: `scale` indicates bound on absolute value of noise.

    Action related:
        mono_init:    initial quantity of monomer;
        mono_density: density of monomer (useful in calculating the volume);
        mono_unit:    unit amount of the "adding monomer" action;
        mono_cap:     maximum quantity (budget) of monomer.
        (Other X_init, X_unit, X_cap variables have similar definitions.)

        addition_noise: `scale` indicates percentage noise in addition unit
                        amount (of any addable).

        By default, any mix of addables can be added at the same time,
        actions are of the form, e.g., (0, 1, 1, 0, 0);

    Rendering related:
        render_figwidth: rendering figure width;
        render_obs:       `None` if no rendering of observation; otherwise
                         a tuple `(plot_name, (start, end))` and the env will
                         plot `observation[start:end]` on `plot_name`

    '''

    metadata = {'render.modes': ['human', 'pygame']}
    addables = MONO, CU1, CU2, DORM1, SOL
    reactants = MONO, CU1, CU2, DORM1

    def __init__(self, max_rad_len=100, termination=True,
                 step_time=1e1, step_time_noise=None,
                 completion_time=1e5, min_steps=100,
                 k_prop=1e4, k_act=1e-2, k_deact=1e5, k_ter=1e10, k_noise=None,
                 obs_noise=None,
                 mono_init=1.0, mono_density=1.0, mono_unit=0.01, mono_cap=None,
                 cu1_init=0.1, cu1_unit=0.01, cu1_cap=None,
                 cu2_init=0.1, cu2_unit=0.01, cu2_cap=None,
                 dorm1_init=0.1, dorm1_unit=0.01, dorm1_cap=None,
                 sol_init=0.0, sol_density=1.0, sol_unit=0.01, sol_cap=0.0,
                 addition_noise=None,
                 render_figwidth=6, render_obs=None,
                 **kwargs):
        # setup the simulation system
        # fundamental properties of the polymerization process
        self.max_rad_len = max_rad_len
        self.termination = termination

        # step related
        self.step_time = step_time
        self.step_time_noise = step_time_noise
        self.completion_time = completion_time
        self.min_steps = min_steps

        # rate constants
        rate_constant = {K_POLY: k_prop, K_ACT: k_act, K_DEACT: k_deact}
        rate_constant[K_TER] = k_ter if termination else 0.0
        self.rate_constant = rate_constant
        self.k_noise = k_noise

        # index (used in self.atrp and self.atrp_jac)
        self.index = self.init_index()

        # initial quant
        self.obs_noise = obs_noise
        add_init = {MONO: mono_init, CU1: cu1_init, CU2: cu2_init,
                    DORM1: dorm1_init, SOL: sol_init}
        self.add_init = add_init
        inv_dens = {MONO: 1. / mono_density, SOL: 1. / sol_density,
                    CU1: 0.0, CU2: 0.0, DORM1: 0.0}
        self.inverse_density = inv_dens
        self.init_volume = sum(add_init[a] * inv_dens[a] for a in self.addables)
        self.init_quant = self.calc_init_quant()

        # actions
        self.add_unit = {MONO: mono_unit, CU1: cu1_unit, CU2: cu2_unit,
                         DORM1: dorm1_unit, SOL: sol_unit}
        self.add_cap = {MONO: mono_cap, CU1: cu1_cap, CU2: cu2_cap,
                        DORM1: dorm1_cap, SOL: sol_cap}

        self.action_space = gym.spaces.MultiBinary(len(self.addables))
        self.addition_noise = addition_noise

        # initialize rewarding scheme (mostly in derived classes)
        self._init_reward(**kwargs)

        # rendering
        figratio = 1.25 if termination else 0.75
        render_figsize = render_figwidth, render_figwidth * figratio
        self.render_figsize = render_figsize
        self.render_obs = render_obs
        self.axes = None

        # ode integrator
        quant_len = len(self.init_quant)
        self.atrp_return = np.zeros(quant_len)
        self.atrp_jac_return = np.zeros([quant_len, quant_len])
        odeint = ode(self.atrp, self.atrp_jac)
        odeint.set_integrator(ODE_INT, method=ODE_METHOD, nsteps=ODE_NSTEPS)
        self.odeint = odeint

    def reset(self):
        self.step_count = 0
        self.added = self.add_init.copy()
        self.last_action = None
        self.quant = self.init_quant.copy()
        self.volume = self.init_volume
        return self.observation()

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.take_action(action)
        done = self.done()
        info = {}
        if self.step_time_noise is not None:
            noise = percent_noise(self.step_time, noise=self.step_time_noise)
            step_time = self.step_time + noise
        else:
            step_time = self.step_time
        run_time = self.completion_time if done else step_time
        k_noise = None if done else self.k_noise

        # run atrp simulation and obtain a new observation
        self.run_atrp(run_time, k_noise)
        observation = self.observation()

        # collect reward
        reward = self._reward(done)

        # compute monomer conversion rate
        mono_idx = self.index[MONO]
        if self.added[mono_idx]:
            mono_conv = 1.0 - self.quant[mono_idx] / self.added[mono_idx]
        else:
            mono_conv = 0.0
        info['mono_conv'] = mono_conv

        return observation, reward, done, info

    def render(self, mode='human'):
        from matplotlib import pyplot as plt, patches as patches
        if mode == 'pygame':
            import matplotlib.backends.backend_agg as agg
            import pygame

        if self.render_obs is not None:
            plot_name, (start, end) = self.render_obs
            obs = self.observation()
            obs_p = obs[start:end]
        if self.axes is None:
            # first time render; build figure and axes
            self.render_fig = plt.figure(figsize=self.render_figsize)
            self.axes = {}
            self.plots = {}
            num_plots = 5 if self.termination else 3
            self.init_plot(DORM, 1, num_plots)
            plt.title('Quantities')
            self.init_plot(RAD, 2, num_plots)
            if self.termination:
                self.init_plot(TER, 3, num_plots)
                stable_chains = self.stable_chains()
                self.init_plot(STABLE, 4, num_plots)
            if self.render_obs is not None:
                len_values = len(obs_p)
                linspace = np.linspace(1, len_values, len_values)
                axis = self.axes[plot_name]
                obs_plot = axis.plot(linspace, obs_p, linestyle='dashed',
                                     color='c', label='Observation')[0]
                axis.legend()
                self.plots['obs'] = obs_plot
            action_axis = plt.subplot(num_plots, 1, num_plots)
            action_axis.get_xaxis().set_visible(False)
            action_axis.get_yaxis().set_visible(False)
            self.action_rect = {}
            self.species_added = {}
            self.species_quant = {}
            action_pos = 0.0, 0.2, 0.4, 0.6, 0.8
            action_labels = 'Monomer', 'Cu(I)', 'Cu(II)', 'Initiator', 'Solvent'
            zip_iter = zip(action_pos, action_labels, self.addables)
            for pos, label, anum in zip_iter:
                color = 'y' if self.capped(anum) else 'r'
                rect = patches.Rectangle((pos, 0.0), 0.18, 1.0,
                                         color=color, fill=True)
                action_axis.add_patch(rect)
                action_axis.annotate(label, (pos + 0.03, 0.7))
                added_label = action_axis.annotate('', (pos + 0.02, 0.4))
                self.action_rect[anum] = rect
                self.species_added[anum] = added_label
            for pos, anum in zip(action_pos[:-1], self.reactants):
                quant_label = action_axis.annotate('', (pos + 0.01, 0.1))
                self.species_quant[anum] = quant_label
            plt.xlabel('Chain length')
            plt.tight_layout()
            if mode == 'pygame':
                # for pygame window in interactive mode
                dpi = self.render_fig.get_dpi()
                resolution = tuple(int(x * dpi) for x in self.render_figsize)
                pygame.display.init()
                pygame.display.set_mode(resolution)
                self.render_screen = pygame.display.get_surface()
            else:
                plt.gcf().show()
        else:
            # not first time render; update
            self.update_plot(DORM)
            self.update_plot(RAD)
            if self.termination:
                self.update_plot(TER)
                self.update_plot(STABLE)
            if self.render_obs is not None:
                self.plots['obs'].set_ydata(obs_p)

        # update button colors
        last_action = self.last_action
        if last_action is None:
            last_action = 0, 0, 0, 0, 0
        for act, anum in zip(last_action, self.addables):
            rect = self.action_rect[anum]
            if self.capped(anum):
                color = 'y'
            else:
                color = 'g' if act else 'r'
            rect.set_color(color)

        # write button labels for addable species
        added = self.added
        add_cap = self.add_cap
        for anum in self.addables:
            this_added = added[anum]
            this_cap = add_cap[anum]
            amount_label = '{:.2f}/{:.2f}'.format(this_added, this_cap)
            self.species_added[anum].set_text(amount_label)
        for anum in self.reactants:
            this_quant = self.quant[self.index[anum]]
            quant_label = 'cur: {:.4f}'.format(this_quant)
            self.species_quant[anum].set_text(quant_label)

        # render to the screen
        if mode == 'pygame':
            canvas = agg.FigureCanvasAgg(self.render_fig)
            canvas.draw()
            raw_data = canvas.get_renderer().tostring_rgb()
            size = canvas.get_width_height()
            surf = pygame.image.fromstring(raw_data, size, 'RGB')
            self.render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        else:
            plt.gcf().canvas.draw()

    def close(self):
        if self.axes is not None:
            self.axes = None
            self.plots = None
            if 'plt' in locals():
                plt.close()

    def init_index(self):
        max_rad_len = self.max_rad_len

        # variable indices (slices) for the ODE solver
        index = {MONO: 0, CU1: 1, CU2: 2, DORM1: 3}
        dorm_from = 3
        index[DORM] = slice(dorm_from, dorm_from + max_rad_len)
        rad_from = 3 + max_rad_len
        index[RAD] = slice(rad_from, rad_from + max_rad_len)

        if self.termination:
            # slices for terminated chains (ter)
            ter_start = 3 + 2 * max_rad_len

            # total number of terminated chains is 2n-1
            index[TER] = slice(ter_start, ter_start + 2 * max_rad_len - 1)

            # slices for length 2 to n and length n+1 to 2n
            index[TER_SLICES] = [slice(None, p) for p in range(1, max_rad_len)]
            index[TER_SLICES].extend(slice(p, None) for p in range(max_rad_len))

        return index

    def calc_init_quant(self):
        max_rad_len = self.max_rad_len
        # observation
        quant_len = 3 + 2 * max_rad_len
        max_chain_len = max_rad_len
        self.rad_chain_lengths = np.arange(1, 1 + max_rad_len)
        if self.termination:
            quant_len += 2 * max_rad_len - 1
            max_chain_len += max_rad_len
            self.ter_chain_lengths = np.arange(2, 1 + max_chain_len)
        self.max_chain_len = max_chain_len

        # obs is added amount of [MONO, CU1, CU2, DORM1, SOL] and self.quant
        obs_len = 5 + quant_len
        self.observation_space = gym.spaces.Box(0.0, np.inf, shape=(obs_len,),
                                                dtype=np.float32)

        # build initial variable
        init_quant = np.zeros(quant_len)
        index = self.index
        add_init = self.add_init
        for key in self.reactants:
            init_quant[index[key]] = add_init[key]
        return init_quant

    def take_action(self, action):
        self.add(action, MONO)
        self.add(action, CU1)
        self.add(action, CU2)
        self.add(action, DORM1)
        self.add(action, SOL, change_quant=False)

    def add(self, action, key, change_quant=True):
        if action[key] and not self.capped(key):
            unit = self.add_unit[key]
            if self.addition_noise is not None:
                noise = percent_noise(unit, self.addition_noise)
                unit = unit + noise
            cap = self.add_cap[key]
            amount = unit if cap is None else min(unit, cap - self.added[key])
            amount = max(amount, 0.0)
            if change_quant:
                self.quant[self.index[key]] += amount
            self.volume += amount * self.inverse_density[key]
            self.added[key] += amount

    def observation(self):
        # added amount isn't noisy
        added = [self.added[key] for key in self.addables]
        current_quant = self.quant.copy()
        # current_quant could be noisy
        if self.obs_noise is not None:
            noise = absolute_noise(self.obs_noise, len(current_quant))
            current_quant += noise
        return np.concatenate([added, current_quant]).astype(np.float32)

    def stable_chains(self):
        quant = self.quant
        index = self.index
        stable_chains = np.zeros(self.max_chain_len)
        stable_chains[:self.max_rad_len] = quant[index[DORM]]
        if self.termination:
            stable_chains[1:] += quant[index[TER]]
        return stable_chains

    def done(self):
        min_steps_exceeded = self.step_count >= self.min_steps
        all_capped = all(self.capped(s) for s in self.addables)
        return min_steps_exceeded and all_capped

    def run_atrp(self, step_time, k_noise):
        # solve atrp odes to get new concentration
        volume = self.volume
        conc = self.quant / volume
        odeint = self.odeint
        odeint.set_initial_value(conc, 0.0)

        # setup a copy of self.rate_constant
        self.rate_constant_use = self.rate_constant.copy()

        # add noise to rate constants if requested
        if k_noise is not None:
            for key, value in self.rate_constant_use.items():
                noise = percent_noise(value, k_noise)
                self.rate_constant_use[key] = value + noise

        # perform integration
        conc = odeint.integrate(step_time)

        # compute and adjust 'quant' so that (monomer + initiator) is conserved
        quant = conc * volume
        index = self.index
        added = self.added
        ref_quant_eq_mono = added[MONO] + added[DORM1]
        mono = quant[index[MONO]]
        dorm = quant[index[DORM]]
        rad = quant[index[RAD]]
        quant_eq_mono = mono + (dorm + rad).dot(self.rad_chain_lengths)
        if self.termination:
            quant_eq_mono += quant[index[TER]].dot(self.ter_chain_lengths)
        ratio = ref_quant_eq_mono / quant_eq_mono if quant_eq_mono else 1.0
        quant *= ratio
        self.quant = quant

    def chain(self, key):
        quant = self.quant
        index = self.index
        if key in [RAD, DORM]:
            chain = quant[index[key]]
        elif key == TER:
            chain = quant[index[TER]]
        elif key == STABLE:
            chain = self.stable_chains()
        return chain

    def capped(self, key):
        unit = self.add_unit[key]
        added_eps = self.added[key] + unit * EPS
        cap = self.add_cap[key]
        return cap is not None and added_eps > cap

    def atrp(self, time, var):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant_use
        k_prop = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_deact = rate_constant[K_DEACT]

        index = self.index
        mono_index = index[MONO]
        cu1_index = index[CU1]
        cu2_index = index[CU2]
        dorm_slice = index[DORM]
        rad_slice = index[RAD]

        mono = var[mono_index]
        cu1 = var[cu1_index]
        cu2 = var[cu2_index]
        dorm = var[dorm_slice]
        rad = var[rad_slice]

        dvar = self.atrp_return

        neg_kp_mono = -k_prop * mono
        sum_rad = rad.sum()

        # monomer
        dvar[mono_index] = neg_kp_mono * sum_rad

        # dormant chains
        dvar_dorm = dvar[dorm_slice]
        dvar_dorm[...] = rad.dot(k_deact * cu2) - dorm.dot(k_act * cu1)

        # radicals
        dvar_rad = dvar[rad_slice]
        neg_kp_mono_rad = rad.dot(neg_kp_mono)
        dvar_rad[...] = neg_kp_mono_rad - dvar_dorm
        dvar_rad[1:] -= neg_kp_mono_rad[:-1]

        # Cu(I)
        sum_dvar_dorm = dvar_dorm.sum()
        dvar[cu1_index] = sum_dvar_dorm

        # Cu(II)
        dvar[cu2_index] = -sum_dvar_dorm

        # terminated chains
        if self.termination:
            kt = rate_constant[K_TER]
            dvar_rad -= rad.dot(2 * kt * sum_rad)
            dvar_ter = dvar[index[TER]]
            for p, ter_slice in enumerate(index[TER_SLICES]):
                rad_part = rad[ter_slice]
                dvar_ter[p] = rad_part.dot(rad_part[::-1])
            dvar_ter *= kt

        return dvar

    def atrp_jac(self, time, var):
        max_rad_len = self.max_rad_len

        rate_constant = self.rate_constant_use
        k_prop = rate_constant[K_POLY]
        k_act = rate_constant[K_ACT]
        k_deact = rate_constant[K_DEACT]

        index = self.index
        mono_index = index[MONO]
        cu1_index = index[CU1]
        cu2_index = index[CU2]
        dorm_slice = index[DORM]
        rad_slice = index[RAD]

        mono = var[mono_index]
        cu1 = var[cu1_index]
        cu2 = var[cu2_index]
        dorm = var[dorm_slice]
        rad = var[rad_slice]

        kp_mono = k_prop * mono
        ka_cu1 = k_act * cu1
        kd_cu2 = k_deact * cu2
        sum_rad = rad.sum()
        ka_dorm = k_act * dorm
        neg_kd_rad = (-k_deact) * rad

        jac = self.atrp_jac_return

        # monomer
        jac_mono = jac[mono_index]
        jac_mono[mono_index] = -k_prop * sum_rad
        jac_mono[rad_slice] = -kp_mono

        # dormant chains
        jac_dorm = jac[dorm_slice]

        # zero jac_dorm_cu1cu2 since it's subject to a -= later
        jac_dorm[:, cu1_index:cu2_index+1] = 0.0

        np.fill_diagonal(jac_dorm[:, dorm_slice], -ka_cu1)
        np.fill_diagonal(jac_dorm[:, rad_slice], kd_cu2)
        jac_dorm[:, cu1_index] -= ka_dorm
        jac_dorm[:, cu2_index] -= neg_kd_rad

        # Cu(I)
        jac_cu1 = jac[cu1_index]
        jac_cu1[cu1_index] = -k_act * dorm.sum()
        jac_cu1[cu2_index] = k_deact * sum_rad
        jac_cu1[dorm_slice] = -ka_cu1
        jac_cu1[rad_slice] = kd_cu2

        # Cu(II)
        jac[cu2_index] = -jac[cu1_index]

        # radicals
        jac_rad = jac[rad_slice]
        jac_rad[:, mono_index] = (-k_prop) * rad
        jac_rad[1:, mono_index] += k_prop * rad[:-1]
        jac_rad[:, cu1_index] = ka_dorm
        jac_rad[:, cu2_index] = neg_kd_rad
        np.fill_diagonal(jac_rad[:, dorm_slice], ka_cu1)
        jac_rad_rad = jac_rad[:, rad_slice]

        # zero jac_rad_rad if termination since it's subject to a -= later
        if self.termination:
            jac_rad_rad[...] = 0.0

        kt2 = 2 * rate_constant[K_TER]
        np.fill_diagonal(jac_rad_rad, -(kp_mono + kd_cu2 + kt2 * sum_rad))
        np.fill_diagonal(jac_rad_rad[1:, :-1], kp_mono)

        # terminated chains
        if self.termination:
            jac_rad_rad -= rad.dot(kt2)[:, np.newaxis]

            # jac_ter_rad[m, n] = 2kt P{n-k}
            jac_ter_rad = jac[index[TER], rad_slice]
            for p, ter_slice in enumerate(index[TER_SLICES]):
                jac_ter_rad[p, ter_slice] = rad[ter_slice][::-1]
            jac_ter_rad *= kt2

        return jac

    def init_plot(self, key, num, num_plots):
        import matplotlib.pyplot as plt
        values = self.chain(key)
        len_values = len(values)
        chain_label_dict = {DORM: 'Dormant chains',
                            RAD: 'Radical chains',
                            TER: 'Terminated chains',
                            STABLE: 'All stable chains'}
        label = chain_label_dict[key]
        axis = plt.subplot(num_plots, 1, num)
        chain_start = 2 if key == TER else 1
        linspace = np.linspace(chain_start, len_values, len_values)
        plot = axis.plot(linspace, values, label=label)[0]
        ymax = np.max(values) * MARGIN_SCALE
        if not ymax:
            ymax = EPS
        axis.set_ylim([0, ymax])
        self._render_reward_init(key, axis)
        axis.legend()
        axis.set_xlim([0, self.max_chain_len])
        self.axes[key] = axis
        self.plots[key] = plot

    def update_plot(self, key):
        values = self.chain(key)
        ymax = np.max(values) * MARGIN_SCALE
        if not ymax:
            ymax = EPS
        axis = self.axes[key]
        axis.set_ylim([0, ymax])
        self._render_reward_update(key, axis)
        self.plots[key].set_ydata(values)

    def _init_reward(self, *args, **kwargs):
        pass

    def _reward(self, *args, **kwargs):
        return 0.0

    def _render_reward_init(self, *args, **kwargs):
        pass

    def _render_reward_update(self, *args, **kwargs):
        pass


