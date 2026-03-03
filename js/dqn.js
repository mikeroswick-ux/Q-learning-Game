// === DQN 智能体 ===

class DQNAgent {
    constructor(numStates, numActions, hyperparams) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.hyperparams = hyperparams;
        
        this.memory = [];
        this.stepCount = 0;
        
        // 检查 tfjs 是否加载
        if(typeof tf === 'undefined') {
            console.error("TensorFlow.js not loaded!");
            return;
        }

        this.model = this.createModel();
        this.targetModel = this.createModel();
        this.updateTargetModel();
        
        console.log("DQN Agent Initialized.");
    }

    createModel() {
        const model = tf.sequential();
        // 隐藏层 1
        model.add(tf.layers.dense({
            units: this.hyperparams.hiddenUnits, 
            inputShape: [this.numStates], 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        // 隐藏层 2
        model.add(tf.layers.dense({
            units: this.hyperparams.hiddenUnits, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        // 输出层
        model.add(tf.layers.dense({
            units: this.numActions, 
            activation: 'linear'
        }));

        model.compile({
            optimizer: tf.train.adam(this.hyperparams.learningRate),
            loss: 'meanSquaredError'
        });
        return model;
    }

    updateTargetModel() {
        const weights = this.model.getWeights();
        const weightsCopies = [];
        for (let i = 0; i < weights.length; i++) {
            weightsCopies[i] = weights[i].clone();
        }
        this.targetModel.setWeights(weightsCopies);
    }

    remember(state, action, reward, nextState, done) {
        if (this.memory.length >= this.hyperparams.memorySize) {
            this.memory.shift();
        }
        this.memory.push({ state, action, reward, nextState, done });
    }

    predict(state) {
        // Epsilon-Greedy 策略
        if (Math.random() < this.hyperparams.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        }
        return tf.tidy(() => {
            const input = tf.tensor2d([state], [1, this.numStates]);
            const output = this.model.predict(input);
            return output.argMax(1).dataSync()[0];
        });
    }

    async replay() {
        if (this.memory.length < this.hyperparams.trainStart) return null;

        this.stepCount++;
        if (this.stepCount % this.hyperparams.syncInterval === 0) {
            this.updateTargetModel();
        }

        const batchSize = this.hyperparams.batchSize;
        const batch = [];
        for(let i=0; i<batchSize; i++){
            const idx = Math.floor(Math.random() * this.memory.length);
            batch.push(this.memory[idx]);
        }

        const lossInfo = await tf.tidy(() => {
            const states = tf.tensor2d(batch.map(x => x.state), [batchSize, this.numStates]);
            const nextStates = tf.tensor2d(batch.map(x => x.nextState), [batchSize, this.numStates]);
            const actions = batch.map(x => x.action);
            const rewards = batch.map(x => x.reward);
            const dones = batch.map(x => x.done);

            const targetNextQs = this.targetModel.predict(nextStates);
            const maxNextQs = targetNextQs.max(1).dataSync();
            
            const qs = this.model.predict(states);
            const qsData = qs.arraySync();

            for (let i = 0; i < batchSize; i++) {
                let target = rewards[i];
                if (!dones[i]) {
                    target += this.hyperparams.gamma * maxNextQs[i];
                }
                qsData[i][actions[i]] = target; 
            }
            
            const targetTensor = tf.tensor2d(qsData, [batchSize, this.numActions]);
            return this.model.trainOnBatch(states, targetTensor);
        });
        
        return lossInfo;
    }
}