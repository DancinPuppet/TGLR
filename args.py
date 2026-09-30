class Args(object):
    def __init__(self):

        # GPU selection
        self.gpu_id = 0

         # Dataset selection
        # self.graph_type = 'twitter25' # diff_weight = 0.1 False start = 0 obs_len = 5
        # self.graph_type = 'twitter15' # diff_weight = 0.01 False start = 0 obs_len = 2
        # self.graph_type = 'twitter16' # diff_weight = 0.01 False start = 0 obs_len = 3
        # self.graph_type = 'weibo' # diff_weight = 0.01 False start = 0 obs_len = 2
        # self.graph_type = 'digg'
        self.graph_type = 'karate_SI'  # diff_weight = 0.1 False start = 0 obs_len = 20 ttt_steps = 100
        # self.graph_type = 'karate_SIR'  # diff_weight = 0.1 False start = 0 obs_len = 20 ttt_steps = 100
        # self.graph_type = 'jazz_SI'    # diff_weight = 0.1 False start = 0 obs_len = 5 ttt_steps = 100
        # self.graph_type = 'jazz_SIR'  # diff_weight = 0.1 False start = 0 obs_len = 5 ttt_steps = 100
        # self.graph_type = 'cora_ml_SI'  # diff_weight = 1.0 start = 0 obs_len = 5 ttt_steps = 30
        # self.graph_type = 'cora_ml_SIR'  # diff_weight = 1.0 start = 0 obs_len = 5 ttt_steps = 30
        # self.graph_type = 'facebook_SI' # diff_weight = 1.0 True start = 0 obs_len = 5 ttt_steps = 30
        # self.graph_type = 'facebook_SIR' # diff_weight = 1.0 True start = 0 obs_len = 5 ttt_steps = 30

        # Maximum number of nodes in the dataset
        self.max_num_node = None

        # Whether to include user profile features
        self.features_with_profiles = False

        # Model variants for ablation experiments
        # # Guidance + Alignment + CFG + Diffusion
        # self.model_name = "GLCFGD"
        # # Guidance + Alignment + Filtering + Diffusion
        # self.model_name = "GLAD"
        # # Guidance + Filtering + Latent + Diffusion
        self.model_name = "TGLR"
        # # Filtering + Diffusion
        # self.model_name = "TGLR_wo_G"
        # # Guidance + Filtering + Diffusion
        # self.model_name = "TGLR_wo_LA"
        # # Guidance + Alignment + Diffusion
        # self.model_name = "TGLR_wo_T"
        # # Guidance + Alignment + Filtering
        # self.model_name = "TGLR_wo_D"
        # # Guidance + Dynamic
        # self.model_name = "TGLR_w_Dy"
        # # Guidance + Latent + Filtering + Diffusion + Concatenate
        # self.model_name = "TGLR_w_C"
        # # Guidance + Latent + Filtering + Diffusion + Concatenate + CFG
        # self.model_name = "TGLR_w_CA"

        # Paths
        self.root_dir = './data/'
        self.data_path = self.root_dir + self.graph_type + '/'
        self.graphs_saved_path = self.root_dir + 'saved_graphs/' + self.graph_type + '_graph.pkl'
        self.save_path = './model_saves/checkpoints'
        self.visual_save_path = './model_saves/visual'
        # Training logs
        self.output_save_path = './output'

        # Module options
        # 1. Whether to order nodes by their scores
        self.scored_based_mapping = False
        self.use_diffusion = True
        self.visual = False
        self.visual_interval = 20
        self.pruning = False
        self.is_sampling = False # Dataset-dependent setting
        self.diff_weight = 0.1 # Dataset-dependent setting
        self.kl_weight = 0.01  # λ_kl
        self.ttt_steps = 100 # Dataset-dependent setting
        # 2. Whether to resume interrupted training
        self.resume_train = False
        if self.resume_train:
            self.checkpoint_path = f'./model_saves/checkpoints/{self.model_name}_{self.graph_type}_checkpoint_epoch_100.pt'
        # 3. Whether to load a trained model directly
        self.load_model = False
        # 4. Whether to use the approximate formulation
        self.approximate = True
        if self.load_model:
            self.load_model_path = f'./model_saves/checkpoints/{self.model_name}_{self.graph_type}_checkpoint_epoch_50.pt'
            self.best_model_path = f'./model_saves/checkpoints/{self.model_name}_best_model.pt'

        # Training parameters
        # Number of data-loading worker processes
        self.num_workers = 0
        self.batch_size = 1
        self.num_batches = 1000
        self.obs_len = 20 # Number of observed time steps; dataset-dependent setting
        self.node_features_dim = 8 + (4 * self.obs_len - 2)  + 2
        self.ob_start_idx = 0 # Observation start index; dataset-dependent setting
        self.threshold = 0.5
        # Initial learning rate
        self.lr = 0.001
        # Learning rate decay
        self.lr_rate = 0.1
        self.epochs = 300
        self.milestones = [100, 200]

        # Model parameters
        # GAT configuration
        self.in_channels = self.node_features_dim
        self.hidden_channels = 32
        self.out_channels = 16
        self.heads = 4
        self.dropout = 0.2

        # # VAE
        self.latent_dim = 16
        self.hidden_dim = 32
        self.node_features_hidden_dim = 32
        # KL weight schedule
        self.kl_start_weight = 0.0001
        self.kl_end_weight = 0.01

        # Distribution alignment
        self.distill_weight = 1.0   # λ_d
        self.student_pred_weight = 0.001
        self.student_hidden_dim = self.latent_dim
        self.student_layers = 2 # l 3
        self.student_use_gate = True
        self.distill_kl_weight = 0.01 # λ_d_kl
        # Prevent exploding gradients
        self.logvar_min = -3.0 # l -6.0
        self.logvar_max = 0.5 # l 2.0


        # # diffusion
        self.beta_start = 1e-4  # Initial noise coefficient; candidate values: 1e-5, 1e-4
        self.beta_end = 0.01 # Final noise coefficient; candidate values: 0.0001, 0.01
        self.max_time_steps = 100
        self.pred_start_weight = 0.5
        self.pred_end_weight = 1.5
        # # # z0 Predictor
        self.prime_predictor_hidden_dim = 32
        self.time_embed_dim = 16

        # Model evaluation
        self.epoch_test = 1
        # Checkpoint interval during training
        self.save_interval = 100
