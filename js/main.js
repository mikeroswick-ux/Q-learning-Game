// === 主程序逻辑 ===

// 1. 初始化 DOM 元素和环境
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const plotCanvas = document.getElementById('plot');
const pCtx = plotCanvas.getContext('2d');

const W = canvas.width;
const H = canvas.height;

// 2. 超参数与全局状态
const ACTIONS = ['left','right','up','down','stay','attack','dodge_left','dodge_right','dodge_up','dodge_down'];
const NUM_ACTIONS = ACTIONS.length;
const NUM_STATES = 9; 

const HYPERPARAMS = {
    learningRate: 0.001,
    gamma: 0.95,
    epsilon: 1.0, 
    epsilonMin: 0.05,
    epsilonDecay: 0.995,
    batchSize: 64,
    memorySize: 20000,
    trainStart: 1000,
    syncInterval: 500,
    hiddenUnits: 64
};

// 游戏状态变量
let player, boss;
let agent;
let timeScale = 1;
let episode = 0;
let running = false;
let training = false;
let battleTimer = 100;
let cumReward = 0;
let episodeRewards = [];

// 临时变量 (上一帧的状态和动作)
let lastStateVec = null;
let lastActionIdx = null;

// Boss 参数
const BOSS_WINDUP = 1.5, BOSS_ACTIVE = 0.3, BOSS_RECOVERY = 1;
const PLAYER_WINDUP = 0.3, PLAYER_ACTIVE = 0.15, PLAYER_RECOVERY = 0.3;

// 分数统计
let scoreBreakdown = {
    survival: 0, proximity: 0, boundary: 0, center: 0, attack_range: 0, 
    hit_boss: 0, hit_by_boss: 0, miss: 0, dodge: 0, terminal: 0
};

// 3. 初始化函数
function initGame() {
    player = new Entity(200, H/2, '#4caf50', true);
    boss = new Entity(W-200, H/2, '#e91e63', false);
    
    // 初始化 Boss 特有属性
    boss.hp = 300;
    boss.currentAction = 'slowApproach';
    boss.actionTimer = 0; boss.actionCooldown = 0;
    boss.actionLockedUntil = 0; boss.lastRetreatTime = -999;
    
    // 初始化 Player 特有属性
    player.dodgeCooldown = 0; player.isDodging = false;
    
    // 如果是第一次运行，初始化 Agent
    if (!agent) {
        agent = new DQNAgent(NUM_STATES, NUM_ACTIONS, HYPERPARAMS);
    }
}

// 4. Boss AI 逻辑 (环境规则)
function bossAI(dt) {
    if(boss.actionTimer > 0) boss.actionTimer -= dt;
    if(boss.actionCooldown > 0) boss.actionCooldown -= dt;
    
    let now = performance.now()/1000;
    let d = dist(boss, player);
    
    // 如果正在攻击，锁定移动
    if(boss.attack.state !== 'idle'){ 
        boss.vx = boss.vy = 0; 
        boss.currentIntent = 'boss_closeAttack'; 
        return; 
    }

    // === 决策阶段 ===
    if(boss.actionTimer <= 0 && boss.actionCooldown <= 0){
        // 锁定检查
        if(boss.actionLockedUntil > now){ 
            boss.actionTimer = 0.2; // 维持一个小的时间片以执行当前动作
            boss.actionCooldown = 0; 
            return; 
        }
        
        // 阈值定义
        const RETREAT_THRESHOLD = 90; // 玩家贴脸距离
        const APPROACH_THRESHOLD = 180; // 追击距离
        const OPTIMAL_DIST = 130; // 理想作战距离
        
        // 边界检查 (阈值60)
        const BOUND_PAD = 60; 
        const distL = boss.x - BOUND_PAD, distR = W - boss.x - BOUND_PAD;
        const distT = boss.y - BOUND_PAD, distB = H - boss.y - BOUND_PAD;
        let nearBound = Math.min(distL, distR, distT, distB) < 0;
        
        // 后撤冷却检查
        let canRetreat = (now - boss.lastRetreatTime) >= 2.0;

        // 优先级 1: 被逼墙角且玩家贴脸 -> 强行突围/回中
        if(d < RETREAT_THRESHOLD && nearBound){
             boss.currentAction = 'returnCenter'; 
             boss.actionTimer = 0.8; 
             boss.actionCooldown = 0.5;
             boss.actionLockedUntil = now + 0.8;
        } 
        // 优先级 2: 单纯靠墙 -> 回中
        else if(nearBound){
            boss.currentAction = 'returnCenter'; 
            boss.actionTimer = 1.0; 
            boss.actionCooldown = 0.5; 
            boss.actionLockedUntil = now + 1.0;
        } 
        // 优先级 3: 玩家贴脸 -> 后撤 
        else if(d < RETREAT_THRESHOLD && canRetreat){
            boss.currentAction = 'retreat'; 
            boss.actionTimer = 0.5; 
            boss.actionCooldown = 1.0; 
            boss.actionLockedUntil = now + 0.5; 
            boss.lastRetreatTime = now;
        } 
        // 优先级 4: 距离过远 -> 快速接近
        else if(d > APPROACH_THRESHOLD){
            boss.currentAction = 'approach'; 
            boss.actionTimer = 1.0; 
            boss.actionCooldown = 0.5; 
            boss.actionLockedUntil = now + 1.0;
        } 
        // 优先级 5: 中距离 -> 缓慢调整
        else {
            boss.currentAction = 'slowApproach'; 
            boss.actionTimer = 0.5; 
            boss.actionCooldown = 0.2; 
            // 这里的 Lock 可以短一点，保持灵活性
            boss.actionLockedUntil = now + 0.5;
        }
    }

    // === 执行阶段 ===
    if(boss.currentAction === 'returnCenter'){
        // 只要没到中心就一直走
        let dc = Math.hypot(boss.x-W/2, boss.y-H/2);
        
        // 如果已经很接近中心了，就提前结束动作，允许AI重新决策
        if(dc < 50){ 
            boss.vx=0; boss.vy=0; 
            boss.actionTimer=0; // 强制结束当前计时
        } else { 
            let a = angleTo(boss, {x:W/2, y:H/2}); 
            boss.vx=Math.cos(a)*250; 
            boss.vy=Math.sin(a)*250; 
        }
        boss.currentIntent = 'boss_returnCenter';
        
    } else if(boss.currentAction === 'approach'){
        boss.currentIntent = 'boss_farApproach'; 
        let a = angleTo(boss, player); 
        boss.vx=Math.cos(a)*180; boss.vy=Math.sin(a)*180; boss.facing=a;
        // 攻击逻辑
        if(boss.attack.state==='idle' && Math.random()<0.06){ 
            let ang=angleTo(boss,player); 
            if(inSector(boss,player,ang,80,120)) boss.startAttack(BOSS_WINDUP,BOSS_ACTIVE,BOSS_RECOVERY,120,80,20,ang); 
        }
        
    } else if(boss.currentAction === 'retreat'){
        boss.currentIntent = 'boss_escape'; 
        let a = angleTo(player, boss); 
        boss.vx=Math.cos(a)*150; boss.vy=Math.sin(a)*150; boss.facing=angleTo(boss,player);
        
        // 动态检查，如果退得够远了(>150px)，立即停止
        if(dist(boss, player) > 150) {
            boss.vx = 0; boss.vy = 0;
            boss.actionTimer = 0; // 提前结束
        }
        
    } else if(boss.currentAction === 'slowApproach'){
        boss.currentIntent = 'boss_farApproach'; 
        let a = angleTo(boss, player); 
        boss.vx=Math.cos(a)*60; boss.vy=Math.sin(a)*60; boss.facing=a;
        if(boss.attack.state==='idle' && Math.random()<0.04){ 
            let ang=angleTo(boss,player); 
            if(inSector(boss,player,ang,80,110)) boss.startAttack(BOSS_WINDUP,BOSS_ACTIVE,BOSS_RECOVERY,110,80,20,ang); 
        }
    } else { 
        boss.vx=boss.vy=0; 
    }
}

// 5. 状态向量化
function getStateVector(){
    let dx = (boss.x - player.x) / W; 
    let dy = (boss.y - player.y) / H;
    let px = player.x / W;
    let py = player.y / H;
    let d = dist(player, boss) / Math.hypot(W, H);
    let cd = (player.attack.state!=='idle') ? 1.0 : 0.0;
    
    let bossStateVal = 0;
    if(boss.attack.state==='windup') bossStateVal = 0.33;
    else if(boss.attack.state==='active') bossStateVal = 0.66;
    else if(boss.attack.state==='recovery') bossStateVal = 1.0;
    
    let bossAction = 0;
    if(boss.currentAction === 'approach') bossAction = 0.25;
    else if(boss.currentAction === 'retreat') bossAction = 0.5;
    else if(boss.currentAction === 'returnCenter') bossAction = 0.75;
    
    let bossProx = (dist(player,boss) < 80) ? 1.0 : 0.0;
    
    return [dx, dy, px, py, d, cd, bossStateVal, bossAction, bossProx];
}

// 6. 玩家动作执行
function playerChooseAction() {
    if(player.isDodging){
        player.currentIntent = 'player_dodge_running';
        lastStateVec = null; lastActionIdx = null; 
        return;
    }

    let s = getStateVector(); 
    let actionIdx = agent.predict(s);
    let a = ACTIONS[actionIdx];

    player.currentIntent = `player_${a}`;
    
    let speed = 160; 
    player.vx = player.vy = 0;
    
    // 移动逻辑
    if(a==='left') player.vx = -speed;
    else if(a==='right') player.vx = speed;
    else if(a==='up') player.vy = -speed;
    else if(a==='down') player.vy = speed;
    
    // 攻击逻辑
    else if(a==='attack' && player.attack.state==='idle'){
        let ang = angleTo(player,boss); let r=65, arc=90;
        if(inSector(player,boss,ang,arc,r)){ 
            player.startAttack(PLAYER_WINDUP,PLAYER_ACTIVE,PLAYER_RECOVERY,r,arc,10, ang); 
        } else { 
            player.vx = Math.cos(ang)*speed; player.vy = Math.sin(ang)*speed; a = 'stay'; 
        }
    }
    
    // 闪避逻辑
    if(a.startsWith('dodge_') && player.dodgeCooldown <= 0){
        const DODGE_SPEED = 300, DODGE_TIME = 80/300;
        player.dodgeTimer = DODGE_TIME; player.isDodging = true; 
        
        let da = 0;
        if(a === 'dodge_left') da = Math.PI;
        else if(a === 'dodge_right') da = 0;
        else if(a === 'dodge_up') da = Math.PI / 2;
        else if(a === 'dodge_down') da = -Math.PI / 2;
        
        player.vx = Math.cos(da)*DODGE_SPEED; player.vy = Math.sin(da)*DODGE_SPEED;
    }

    // 面向
    if(a!=='attack' && !a.startsWith('dodge_')){
        if(player.vx!==0||player.vy!==0) player.facing = Math.atan2(player.vy,player.vx);
        else player.facing = angleTo(player,boss);
    } else if(a==='attack') player.facing = angleTo(player,boss);

    lastStateVec = s;
    lastActionIdx = actionIdx;
}

// 7. 奖励计算与训练
function processRewardAndTrain(scoreComps, isTerminal) {
    let reward = 0;
    
    // 细分奖励
    let survivalReward = 0.02; scoreBreakdown.survival += survivalReward; reward += survivalReward;
    
    let curDist = dist(player,boss);
    let proximityReward = 0;
    if(curDist < 50) proximityReward = -0.5;
    else if(curDist < 100) proximityReward = 0.5 + (100 - curDist) * 0.02;
    else if(curDist < 200) proximityReward = 0.5 - (curDist - 100) * 0.005;
    else proximityReward = -0.5;
    scoreBreakdown.proximity += proximityReward; reward += proximityReward;

    let boundaryReward = 0;
    if(player.x < 120) boundaryReward -= (120-player.x)*0.02;
    if(player.x > W-120) boundaryReward -= (player.x-(W-120))*0.02;
    if(player.y < 120) boundaryReward -= (120-player.y)*0.02;
    if(player.y > H-120) boundaryReward -= (player.y-(H-120))*0.02;
    scoreBreakdown.boundary += boundaryReward; reward += boundaryReward;

    if(player.x >= 120 && player.x <= W-120 && player.y >= 120 && player.y <= H-120) {
        scoreBreakdown.center += 0.2; reward += 0.2;
    }

    if(curDist < 120 && player.attack.state === 'idle') {
        scoreBreakdown.attack_range += 0.5; reward += 0.5;
    }

    scoreBreakdown.hit_boss += scoreComps.hit_boss;
    scoreBreakdown.hit_by_boss += scoreComps.hit_by_boss;
    reward += scoreComps.hit_boss + scoreComps.hit_by_boss;
    if(scoreComps.dodge) { scoreBreakdown.dodge += scoreComps.dodge; reward += scoreComps.dodge; }
    if(scoreComps.miss) { scoreBreakdown.miss += scoreComps.miss; reward += scoreComps.miss; }

    let terminalReward = 0;
    if(boss.hp<=0) terminalReward += 1000;
    if(player.hp<=0) terminalReward -= 20;
    if(battleTimer <= 0 && player.hp > 0) terminalReward += 20;
    scoreBreakdown.terminal += terminalReward;
    reward += terminalReward;

    cumReward += reward;

    if(lastStateVec !== null && lastActionIdx !== null){
        let nextStateVec = getStateVector();
        agent.remember(lastStateVec, lastActionIdx, reward, nextStateVec, isTerminal);
    }

    if(training){
        agent.replay();
    }
}

// 8. 游戏循环
let lastTime = performance.now();
let prevPlayerAttackState = 'idle';

function loop(now) {
    let dt = ((now - lastTime) / 1000) * timeScale;
    lastTime = now;

    if (running) {
        playerChooseAction();
        bossAI(dt);
        player.step(dt);
        boss.step(dt);

        // 闪避冷却与状态恢复
        if(player.dodgeCooldown > 0) player.dodgeCooldown = Math.max(0, player.dodgeCooldown - dt);
        if(player.isDodging && player.dodgeTimer === 0){ 
            player.isDodging = false; 
            player.dodgeCooldown = 1.0; 
            player.currentIntent = 'player_stay'; 
        }

        // 碰撞分离
        let d = dist(player, boss), minDist = 24;
        if(d < minDist){
            let angle = angleTo(boss, player), overlap = minDist - d;
            player.x += Math.cos(angle)*overlap; player.y += Math.sin(angle)*overlap;
            player.x = clamp(player.x,20,W-20); player.y = clamp(player.y,20,H-20);
        }

        // 判定与得分
        let scoreComps = { hit_boss:0, hit_by_boss:0, miss:0, dodge:0 };
        // Boss 命中检测
        if(boss.attack.state==='active' && !boss.attack.hasHit){
            if(inSector(boss,player,boss.attack.angle,boss.attack.arc,boss.attack.range)){
                player.hp -= boss.attack.damage; boss.attack.hasHit = true; scoreComps.hit_by_boss = -50;
            }
        }
        // Player 命中检测
        if(player.attack.state==='active' && !player.attack.hasHit){
            if(inSector(player,boss,player.attack.angle,player.attack.arc,player.attack.range)){
                boss.hp -= player.attack.damage; player.attack.hasHit = true; scoreComps.hit_boss = 40;
            }
        }
        // Miss 检测
        if(prevPlayerAttackState==='active' && player.attack.state==='recovery' && !player.attack.hasHit){
            scoreComps.miss = -5;
        }
        // Dodge 检测
        if(boss.attack.state === 'active' && !inSector(boss, player, boss.attack.angle, boss.attack.arc, boss.attack.range)){
            scoreComps.dodge = 3;
        }
        prevPlayerAttackState = player.attack.state;

        battleTimer -= dt;
        let isTerminal = (boss.hp<=0 || player.hp<=0 || battleTimer <= 0);

        processRewardAndTrain(scoreComps, isTerminal);

        if(isTerminal){
            episode++;
            episodeRewards.push(cumReward);
            
            if(HYPERPARAMS.epsilon > HYPERPARAMS.epsilonMin){
                HYPERPARAMS.epsilon *= HYPERPARAMS.epsilonDecay;
            }
            
            // 记录日志
            if(episode % 10 === 0){
                let start = Math.max(0, episode-10); let slice = episodeRewards.slice(start, episode);
                let avg = slice.reduce((a,b)=>a+b,0)/slice.length;
                console.log(`Ep ${episode} | Avg Rw: ${avg.toFixed(1)} | Eps: ${HYPERPARAMS.epsilon.toFixed(3)} | Mem: ${agent.memory.length}`);
                if(window.gc) window.gc(); 
            }

            // 如果在训练，重置环境
            if(training){
                initGame(); 
                cumReward=0; battleTimer=100;
                scoreBreakdown = {survival: 0, proximity: 0, boundary: 0, center: 0, attack_range: 0, hit_boss: 0, hit_by_boss: 0, miss: 0, dodge: 0, terminal: 0};
                lastStateVec = null; lastActionIdx = null;
            } else {
                running = false; 
                document.getElementById('startBtn').textContent = '▶ 开始';
            }
            
            // 更新图表
            let plotAfter = parseInt(document.getElementById('plotAfter').value||100);
            if(episode >= plotAfter) drawPlot();
        }

        render();
        updateScoreBoard();
    }
    requestAnimationFrame(loop);
}

// 9. 渲染函数
function render() {
    ctx.clearRect(0,0,W,H);
    // 网格
    ctx.strokeStyle='rgba(255,255,255,0.03)'; 
    for(let x=0;x<W;x+=40){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke() } 
    for(let y=0;y<H;y+=40){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke() }
    
    // 绘制攻击扇区
    if(boss.attack.state==='windup' || boss.attack.state==='active'){
        let a = (typeof boss.attack.angle==='number')? boss.attack.angle : boss.facing; 
        let alpha = boss.attack.state==='windup'?0.06:0.12; 
        drawAttackSector(ctx,boss,a,boss.attack.arc,boss.attack.range,alpha);
    }
    if(player.attack.state==='windup' || player.attack.state==='active'){
        let a = (typeof player.attack.angle==='number')? player.attack.angle : player.facing; 
        let alpha = player.attack.state==='windup'?0.05:0.1; 
        drawAttackSector(ctx,player,a,player.attack.arc,player.attack.range,alpha);
    }
    
    boss.draw(ctx); player.draw(ctx);
    
    // 文字信息
    ctx.fillStyle='#fff'; ctx.font='14px monospace'; 
    ctx.fillText('Player HP: '+Math.max(0,Math.round(player.hp)),10,18); 
    ctx.fillText('Boss HP: '+Math.max(0,Math.round(boss.hp)),10,36);
    
    ctx.font='11px monospace'; 
    ctx.fillStyle='#999'; ctx.fillText('Player Intent:', 10, 55);
    ctx.fillStyle = intentColors[player.currentIntent] || '#fff'; 
    ctx.fillText(player.currentIntent.replace('player_',''), 100, 55);
    
    ctx.fillStyle='#999'; ctx.fillText('Boss Intent:', 250, 55);
    ctx.fillStyle = intentColors[boss.currentIntent] || '#fff'; 
    ctx.fillText(boss.currentIntent.replace('boss_',''), 330, 55);
    
    document.getElementById('stats').textContent = `Episode: ${episode} | Mem: ${agent.memory.length} | Eps: ${HYPERPARAMS.epsilon.toFixed(3)} | Time: ${Math.max(0, battleTimer).toFixed(1)}s`;
}

function updateScoreBoard(){
    let total = Object.values(scoreBreakdown).reduce((a,b)=>a+b,0);
    let tbody = document.getElementById('scoreBody');
    let labels = [['生存奖励','survival','#4caf50'],['靠近奖励','proximity','#3498db'],['边界惩罚','boundary','#e74c3c'],['中心区奖励','center','#2ecc71'],['攻击范围奖励','attack_range','#f39c12'],['命中得分','hit_boss','#9c27b0'],['被击中扣分','hit_by_boss','#e91e63'],['未中扣分','miss','#95a5a6'],['成功闪避','dodge','#00bcd4'],['终止奖励','terminal','#1abc9c']];
    let html = '';
    labels.forEach(([label, key, color]) => {
        let value = scoreBreakdown[key];
        let percent = total !== 0 ? ((value/total)*100).toFixed(1) : '0.0';
        html += `<tr style="border-bottom:1px solid rgba(76,175,80,0.2)"><td style="padding:4px 0;color:${color}">${label}</td><td style="text-align:right;padding:4px 0">${value.toFixed(2)}</td><td style="text-align:right;padding:4px 0;color:${color}">${total>0?percent:'-'}%</td></tr>`;
    });
    tbody.innerHTML = html;
    document.getElementById('totalScore').textContent = total.toFixed(2);
    document.getElementById('episodeLabel').textContent = `Episode ${episode}`;
}

function drawPlot(){
    let arr = episodeRewards; if(arr.length===0) return;
    let w = plotCanvas.width, h = plotCanvas.height; 
    pCtx.clearRect(0,0,w,h);
    pCtx.strokeStyle='#2b7'; pCtx.lineWidth=1; 
    pCtx.beginPath(); pCtx.moveTo(40,10); pCtx.lineTo(40,h-20); pCtx.lineTo(w-10,h-20); pCtx.stroke();
    
    let max = Math.max(...arr), min = Math.min(...arr); 
    if(max===min){max=min+1}
    let plotW = w-60, plotH = h-40;
    
    pCtx.strokeStyle='#7fd'; pCtx.beginPath();
    arr.forEach((v,i)=>{
        let x = 40 + (i/(arr.length-1 || 1))*plotW;
        let y = 10 + (1 - (v-min)/(max-min))*plotH;
        if(i===0) pCtx.moveTo(x,y); else pCtx.lineTo(x,y);
    });
    pCtx.stroke();
}

// 10. 事件监听与初始化
document.addEventListener('DOMContentLoaded', () => {
    initGame();
    render(); // 初始绘制
    requestAnimationFrame(loop); // 启动循环监听

    // 按钮绑定
    document.getElementById('startBtn').onclick = function(){ 
        running = !running; 
        this.textContent = running? '⏸ 暂停' : '▶ 开始'; 
        if(running){ lastTime=performance.now(); }
    };

    document.getElementById('trainBtn').onclick = function(){ 
        training = !training; 
        this.textContent = training? '⏹ 停止训练' : '⚙ 训练'; 
        if(training){ 
            running=true; 
            document.getElementById('startBtn').textContent = '⏸ 暂停';
            lastTime=performance.now(); 
        }
    };

    document.getElementById('resetBtn').onclick = function(){ 
        // 彻底重置
        episode=0; episodeRewards=[]; cumReward=0; 
        agent = new DQNAgent(NUM_STATES, NUM_ACTIONS, HYPERPARAMS);
        HYPERPARAMS.epsilon = 1.0;
        initGame();
        scoreBreakdown = {survival: 0, proximity: 0, boundary: 0, center: 0, attack_range: 0, hit_boss: 0, hit_by_boss: 0, miss: 0, dodge: 0, terminal: 0};
        render();
        updateScoreBoard();
    };

    document.getElementById('showPlotBtn').onclick = drawPlot;

    // 速度控制
    document.querySelectorAll('.speedBtn').forEach(b=>{ 
        b.addEventListener('click', ()=>{ 
            timeScale = parseFloat(b.dataset.speed);
            document.getElementById('speedLabel').textContent = timeScale + 'x'; 
            document.querySelectorAll('.speedBtn').forEach(btn => btn.classList.remove('active'));
            b.classList.add('active');
        }) 
    });

    // 模型存取
    document.getElementById('saveSnapshotBtn').onclick = async function(){ 
        await agent.model.save('localstorage://my-dqn-model');
        alert(`模型已保存到浏览器 LocalStorage.\n时间: ${new Date().toLocaleTimeString()}`);
    };
    
    document.getElementById('loadSnapshotBtn').onclick = async function(){ 
        try {
            const model = await tf.loadLayersModel('localstorage://my-dqn-model');
            agent.model = model;
            agent.model.compile({optimizer: tf.train.adam(HYPERPARAMS.learningRate), loss: 'meanSquaredError'});
            agent.updateTargetModel();
            alert("模型加载成功！");
        } catch(e) { 
            alert("未找到已保存的模型！"); 
        }
    };

});
