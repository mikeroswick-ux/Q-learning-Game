// === 游戏实体与渲染 ===

// 全局常量 (需要在 main.js 中初始化 Canvas 后赋值，或者直接写死)
const CANVAS_WIDTH = 900;
const CANVAS_HEIGHT = 500;

class Entity {
    constructor(x, y, color, isPlayer) {
        this.x = x; this.y = y;
        this.vx = 0; this.vy = 0;
        this.size = 24;
        this.color = color;
        this.facing = 0;
        this.hp = 100;
        // 攻击状态机: idle -> windup -> active -> recovery
        this.attack = { state: 'idle', timer: 0, windup: 0, active: 0, recovery: 0, range: 110, arc: 90, damage: 10, hasHit: false };
        this.dodgeTimer = 0;
        this.currentIntent = isPlayer ? 'player_stay' : 'boss_cooldown';
        this.isPlayer = isPlayer || false;
    }

    step(dt) {
        // 物理移动
        this.x += this.vx * dt;
        this.y += this.vy * dt;
        // 边界限制
        this.x = clamp(this.x, 20, CANVAS_WIDTH - 20);
        this.y = clamp(this.y, 20, CANVAS_HEIGHT - 20);

        // 攻击状态更新
        if (this.attack.state === 'windup') {
            this.attack.timer -= dt;
            if (this.attack.timer <= 0) {
                this.attack.state = 'active';
                this.attack.timer = this.attack.active;
                this.attack.hasHit = false;
            }
        } else if (this.attack.state === 'active') {
            this.attack.timer -= dt;
            if (this.attack.timer <= 0) {
                this.attack.state = 'recovery';
                this.attack.timer = this.attack.recovery;
            }
        } else if (this.attack.state === 'recovery') {
            this.attack.timer -= dt;
            if (this.attack.timer <= 0) {
                this.attack.state = 'idle';
                this.attack.timer = 0;
                this.attack.hasHit = false;
            }
        }

        // 闪避计时器更新
        if (this.dodgeTimer > 0) {
            this.dodgeTimer = Math.max(0, this.dodgeTimer - dt);
        }
    }

    draw(ctx) {
        let color = intentColors[this.currentIntent] || this.color;
        ctx.fillStyle = color;
        if (this.isPlayer) {
            // 画圆 (玩家)
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size / 2, 0, 2 * Math.PI);
            ctx.fill();
        } else {
            // 画方 (Boss)
            ctx.fillRect(this.x - this.size / 2, this.y - this.size / 2, this.size, this.size);
        }
    }

    startAttack(windup, active, recovery, range, arc, damage, angle) {
        this.attack.windup = windup;
        this.attack.active = active;
        this.attack.recovery = recovery;
        this.attack.range = range;
        this.attack.arc = arc;
        this.attack.damage = damage;
        this.attack.state = 'windup';
        this.attack.timer = windup;
        this.attack.hasHit = false;
        this.attack.angle = (typeof angle === 'number') ? angle : this.facing;
    }
}

// 扇形攻击范围绘制
function drawAttackSector(ctx, ent, angle, arcDeg, radius, alpha) {
    ctx.save();
    ctx.translate(ent.x, ent.y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    // 负一半到正一半
    ctx.arc(0, 0, radius, -arcDeg / 2 * Math.PI / 180, arcDeg / 2 * Math.PI / 180);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,' + alpha + ')';
    ctx.fill();
    ctx.restore();
}

// 扇形判定检测
function inSector(attacker, target, angle, arcDeg, radius) {
    let d = dist(attacker, target);
    if (d > radius) return false;
    let a = angleTo(attacker, target);
    let diff = Math.atan2(Math.sin(a - angle), Math.cos(a - angle));
    return Math.abs(diff) <= (arcDeg / 2) * Math.PI / 180;
}