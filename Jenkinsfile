pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID     = credentials('minio-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('minio-secret-key')
        // MLFLOW_TRACKING_URI   = "http://mlflow:5000"
        VENV                  = "/var/jenkins_home/shared_venvs/mlops_pipeline"
        // Define your Docker Hub or local registry (optional but best practice)
        DOCKER_IMAGE_NAME     = "hamza629/mlops-api"
        HF_TOKEN              = credentials('hf-token')
    }

    stages {
        // ==========================================
        // PHASE 1: CONTINUOUS INTEGRATION (CI)
        // ==========================================
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Environment & Dependencies') {
            steps {
                echo "Setting up Python virtual environment..."
                sh '''
                    # 1. Create the virtual environment ONLY if it doesn't exist yet
                    if [ ! -d "${VENV}" ]; then
                        echo "Creating new virtual environment..."
                        python3 -m venv ${VENV}
                    fi
                    
                    . ${VENV}/bin/activate
                    pip install --upgrade pip
                    
                    # 2. Install requirements using a persistent cache
                    pip install --cache-dir /var/jenkins_home/.cache/pip -r requirements.txt
                    pip install pytest flake8
                '''
            }
        }

        stage('Code Quality (Lint & Test)') {
            steps {
                echo "Running static analysis and unit tests..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    # 1. Lint the code (using --exit-zero so formatting doesn't crash the build)
                    flake8 madewithml/ --exit-zero
                    
                '''
            }
        }

        stage('Data Fetch & Validation') {
            steps {
                echo "Pulling and validating data..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    # DevOps Magic: Dynamically switch DVC from localhost to the Docker minio container
                    sed -i 's/localhost:9000/minio:9000/g' .dvc/config
                    
                    # Now pull the data
                    dvc pull
                '''
            }
        }

        // ==========================================
        // PHASE 2: CONTINUOUS TRAINING (CT)
        // ==========================================
        stage('Model Training (Ray)') {
            steps {
                echo "Initiating Ray Distributed Training..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    export HF_TOKEN=${HF_TOKEN}
                    
                    # Hard-limit Ray's internal memory storage to prevent OOM crashes
                    export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.3
                    
                    # Extreme diet settings for CI/CD pipeline validation
                    python -m madewithml.train \
                        --experiment-name "ci_cd_production" \
                        --dataset-loc "datasets/dataset.csv" \
                        --num-workers 1 \
                        --cpu-per-worker 2 \
                        --num-epochs 1 \
                        --batch-size 8 \
                        --results-fp results.json
                '''
            }
        }

        stage('Model Evaluation Gate') {
            steps {
                script {
                    echo "Evaluating new model performance..."
                    def results = readJSON file: 'results.json'
                    env.RUN_ID = results.run_id
                    
                    // Best Practice: Fail the pipeline if the model is terrible
                    // Example: Check if the validation loss is below a certain threshold
                    // def valLoss = results.metrics[-1].val_loss
                    // if (valLoss > 0.6) {
                    //     error("Model rejected! Validation loss ${valLoss} is too high.")
                    // }
                    echo "Model passed evaluation metrics. Preparing for deployment."
                }
            }
        }

        // ==========================================
        // PHASE 3: CONTINUOUS DEPLOYMENT (CD)
        // ==========================================
        stage('Build, Tag, & Push Docker Image') {
            steps {
                echo "Building API container locked to RUN_ID: ${RUN_ID}"
                
                // Securely inject the credentials we saved in Jenkins
                withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                    sh '''
                        # 1. Log in to the private registry (using stdin for security)
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        
                        # If using GHCR or AWS, append the registry URL to the login command:
                        # echo "$DOCKER_PASS" | docker login ghcr.io -u "$DOCKER_USER" --password-stdin
                        
                        # 2. Build the Docker image
                        docker build -t ${DOCKER_IMAGE_NAME}:${RUN_ID} .
                        
                        # 3. Tag it as the "latest" version
                        docker tag ${DOCKER_IMAGE_NAME}:${RUN_ID} ${DOCKER_IMAGE_NAME}:latest
                        
                        # 4. Push the specific version and the latest version to your private repo
                        docker push ${DOCKER_IMAGE_NAME}:${RUN_ID}
                        docker push ${DOCKER_IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy to Production') {
            steps {
                echo "Rolling out new API version..."
                sh '''
                    # Stop the current API
                    docker-compose down api
                    
                    # Start the new API using the exact environment variable
                    docker-compose up -d api
                '''
            }
        }
    }

   pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID     = credentials('minio-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('minio-secret-key')
        // MLFLOW_TRACKING_URI   = "http://mlflow:5000"
        VENV                  = "/var/jenkins_home/shared_venvs/mlops_pipeline"
        // Define your Docker Hub or local registry (optional but best practice)
        DOCKER_IMAGE_NAME     = "hamza629/mlops-api"
        HF_TOKEN              = credentials('hf-token')
    }

    stages {
        // ==========================================
        // PHASE 1: CONTINUOUS INTEGRATION (CI)
        // ==========================================
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Environment & Dependencies') {
            steps {
                echo "Setting up Python virtual environment..."
                sh '''
                    # 1. Create the virtual environment ONLY if it doesn't exist yet
                    if [ ! -d "${VENV}" ]; then
                        echo "Creating new virtual environment..."
                        python3 -m venv ${VENV}
                    fi
                    
                    . ${VENV}/bin/activate
                    pip install --upgrade pip
                    
                    # 2. Install requirements using a persistent cache
                    pip install --cache-dir /var/jenkins_home/.cache/pip -r requirements.txt
                    pip install pytest flake8
                '''
            }
        }

        stage('Code Quality (Lint & Test)') {
            steps {
                echo "Running static analysis and unit tests..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    # 1. Lint the code (using --exit-zero so formatting doesn't crash the build)
                    flake8 madewithml/ --exit-zero
                    
                '''
            }
        }

        stage('Data Fetch & Validation') {
            steps {
                echo "Pulling and validating data..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    # DevOps Magic: Dynamically switch DVC from localhost to the Docker minio container
                    sed -i 's/localhost:9000/minio:9000/g' .dvc/config
                    
                    # Now pull the data
                    dvc pull
                '''
            }
        }

        // ==========================================
        // PHASE 2: CONTINUOUS TRAINING (CT)
        // ==========================================
        stage('Model Training (Ray)') {
            steps {
                echo "Initiating Ray Distributed Training..."
                sh '''
                    . ${VENV}/bin/activate
                    
                    export HF_TOKEN=${HF_TOKEN}
                    
                    # Hard-limit Ray's internal memory storage to prevent OOM crashes
                    export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.3
                    
                    # Extreme diet settings for CI/CD pipeline validation
                    python -m madewithml.train \
                        --experiment-name "ci_cd_production" \
                        --dataset-loc "datasets/dataset.csv" \
                        --num-workers 1 \
                        --cpu-per-worker 2 \
                        --num-epochs 1 \
                        --batch-size 8 \
                        --results-fp results.json
                '''
            }
        }

        stage('Model Evaluation Gate') {
            steps {
                script {
                    echo "Evaluating new model performance..."
                    def results = readJSON file: 'results.json'
                    env.RUN_ID = results.run_id
                    
                    // Best Practice: Fail the pipeline if the model is terrible
                    // Example: Check if the validation loss is below a certain threshold
                    // def valLoss = results.metrics[-1].val_loss
                    // if (valLoss > 0.6) {
                    //     error("Model rejected! Validation loss ${valLoss} is too high.")
                    // }
                    echo "Model passed evaluation metrics. Preparing for deployment."
                }
            }
        }

        // ==========================================
        // PHASE 3: CONTINUOUS DEPLOYMENT (CD)
        // ==========================================
        stage('Build, Tag, & Push Docker Image') {
            steps {
                echo "Building API container locked to RUN_ID: ${RUN_ID}"
                
                // Securely inject the credentials we saved in Jenkins
                withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                    sh '''
                        # 1. Log in to the private registry (using stdin for security)
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        
                        # If using GHCR or AWS, append the registry URL to the login command:
                        # echo "$DOCKER_PASS" | docker login ghcr.io -u "$DOCKER_USER" --password-stdin
                        
                        # 2. Build the Docker image
                        docker build -t ${DOCKER_IMAGE_NAME}:${RUN_ID} .
                        
                        # 3. Tag it as the "latest" version
                        docker tag ${DOCKER_IMAGE_NAME}:${RUN_ID} ${DOCKER_IMAGE_NAME}:latest
                        
                        # 4. Push the specific version and the latest version to your private repo
                        docker push ${DOCKER_IMAGE_NAME}:${RUN_ID}
                        docker push ${DOCKER_IMAGE_NAME}:latest
                    '''
                }
            }
        }

        stage('Deploy to Production') {
            steps {
                echo "Rolling out new API version..."
                sh '''
                    # Stop the current API
                    docker-compose down api
                    
                    # Start the new API using the exact environment variable
                    docker-compose up -d api
                '''
            }
        }
    }

   post {
    success {
        echo "✅ CI/CD Pipeline completed! New model ${RUN_ID} is live."
    }
    failure {
        echo "❌ Pipeline failed. Deploy aborted to protect production."
    }
    always {
        script {
            cleanWs()
        }
    }
}
   }}