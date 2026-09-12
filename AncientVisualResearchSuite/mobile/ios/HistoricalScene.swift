import Foundation

public struct Evidence: Codable { public let id:String; public let source:String; public let kind:String; public let confidence:Double }
public struct CharacterAgent: Codable { public let id:String; public let role:String; public var x:Double; public var y:Double; public let heading:Double; public let activity:String }
public struct HistoricalScene: Codable {
    public let id:String; public let title:String; public let latitude:Double; public let longitude:Double
    public var characters:[CharacterAgent]; public var evidence:[Evidence]
    public func simulate(seconds:Double, step:Double=0.25) -> [[CharacterAgent]] {
        precondition(seconds >= 0 && step > 0)
        var frames:[[CharacterAgent]]=[]; var t=0.0
        while t <= seconds + 1e-9 {
            frames.append(characters.map { c in
                let active=(c.activity == "idle" || c.activity == "sleep") ? 0.0 : 0.5
                let phase=c.heading * .pi / 180.0 + t * 0.15
                return CharacterAgent(id:c.id,role:c.role,x:c.x + cos(phase)*active*t,y:c.y + sin(phase)*active*t,heading:c.heading,activity:c.activity)
            })
            t += step
        }
        return frames
    }
}
